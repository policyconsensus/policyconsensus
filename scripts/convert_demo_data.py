if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import numpy as np
import click
import zarr
import torch
import torchvision.transforms as T
import cv2
import shutil
import time
from pathlib import Path

from modular_policy.common.replay_buffer import ReplayBuffer
from modular_policy.common.input_util import wait_user_input
from modular_policy.real_world.real_util import real_obs_to_policy_obs, real_obs_to_policy_obs_batched

from typing import Tuple


def _flatten_zarr_structure(old_buffer_path: str):
    """
    Flatten nested zarr structure using Python operations instead of shell commands.
    This is faster and more reliable than using os.system().
    """
    obs_path = Path(old_buffer_path) / "data" / "observations"
    images_path = Path(old_buffer_path) / "data" / "images"
    data_path = Path(old_buffer_path) / "data"
    
    if obs_path.exists():
        print("Moving observations to root level...")
        # Move all files from observations to data root
        for item in obs_path.iterdir():
            if item.is_dir():
                # Move directory contents
                dest_dir = data_path / item.name
                if dest_dir.exists():
                    shutil.rmtree(dest_dir)
                shutil.move(str(item), str(dest_dir))
            else:
                # Move individual files
                dest_file = data_path / item.name
                if dest_file.exists():
                    dest_file.unlink()
                shutil.move(str(item), str(dest_file))
        
        # Handle nested images if they exist
        if images_path.exists():
            print("Moving images to root level...")
            for item in images_path.iterdir():
                dest_path = data_path / item.name
                if dest_path.exists():
                    if dest_path.is_dir():
                        shutil.rmtree(dest_path)
                    else:
                        dest_path.unlink()
                shutil.move(str(item), str(dest_path))
            
            # Remove empty images directory
            try:
                images_path.rmdir()
            except OSError:
                pass  # Directory not empty, that's fine
        
        # Remove empty observations directory
        try:
            obs_path.rmdir() 
        except OSError:
            pass  # Directory not empty, that's fine


@click.command()
@click.option('-d', '--data_dir', required=True, type=str)
@click.option('-s', '--save_path', type=str, default=None)
@click.option('-v', '--vis_size', type=str, default="96x128")
@click.option('--generate_video', is_flag=True, help="Generate videos for inspection")
@click.option('--batch_size', type=int, default=2, help="Number of episodes to process in batch for better performance")
@click.option('-o', '--keys', type=str, multiple=True, default=(
    'camera_0_color',
    'camera_1_color', 
    'camera_2_color',
    'left_tactile',
    'right_tactile',
    'robot_joint', 
    'robot_eef_pose',
    'cartesian_action',
    'joint_action',
))
def main(
    data_dir: str,
    save_path: str,
    vis_size: str,
    generate_video: bool,
    batch_size: int,
    keys: Tuple[str],
):
    if data_dir.endswith('/'):
        data_dir = data_dir[:-1]
    assert os.path.exists(data_dir), f"data source {data_dir} does not exist"
    if save_path is None:
        save_path = f"{data_dir}.zarr"
    assert save_path.endswith('.zarr'), f"save path {save_path} should be a zarr file"

    # check if the save path exists
    if os.path.exists(save_path):
        keypress = wait_user_input(
            valid_input=lambda key: key in ['', 'y', 'n'],
            prompt=f"{save_path} already exists. Overwrite? [y/`n`]: ",
            default='n'
        )
        if keypress == 'n':
            print("Abort")
            return
        else:
            os.system(f"rm -rf {save_path}")

    old_buffer_path = f"{data_dir}/replay_buffer.zarr"

    # Flatten nested zarr structure for fast access using Python operations
    print("Flattening nested zarr structure for optimal performance...")
    _flatten_zarr_structure(old_buffer_path)

    # Now use standard ReplayBuffer for fast access
    print("Loading flattened buffer...")
    old_buffer = ReplayBuffer.create_from_path(old_buffer_path)
    
    # Create buffer without compression for faster processing
    print("Creating temporary uncompressed buffer for fast processing...")
    new_buffer = ReplayBuffer.create_empty_zarr()

    print(f"Processing {old_buffer.n_episodes} episodes in batches of {batch_size}...")
    
    # Process episodes in batches for better performance
    total_episodes = old_buffer.n_episodes
    total_batches = (total_episodes + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for batch_idx, batch_start in enumerate(range(0, total_episodes, batch_size)):
        batch_start_time = time.time()
        batch_end = min(batch_start + batch_size, total_episodes)
        
        # Load batch episodes with minimal copying
        batch_episodes = []
        for i in range(batch_start, batch_end):
            eps = old_buffer.get_episode(i, copy=False)  # Use views when possible
            # Select keys that exist (create shallow copy only for key filtering)
            eps = {k: eps[k] for k in keys if k in eps}
            batch_episodes.append(eps)
        
        # Batch process observations for better performance
        h, w = map(int, vis_size.split('x'))
        
        # Check memory usage - if batch is too large, fall back to single processing
        avg_episode_length = sum(len(list(ep.values())[0]) for ep in batch_episodes) / len(batch_episodes)
        estimated_memory_gb = len(batch_episodes) * avg_episode_length * 3 * (h * w * 3) / (1024**3)
        
        if len(batch_episodes) > 1 and estimated_memory_gb < 4.0:  # Only batch if under 4GB
            # Use batched processing for multiple episodes with fast resize
            processed_batch = real_obs_to_policy_obs_batched(batch_episodes, vis_size=(h, w), use_fast_resize=True)
        else:
            # Fall back to single processing to avoid memory issues
            processed_batch = []
            for ep in batch_episodes:
                processed_ep = real_obs_to_policy_obs(ep, vis_size=(h, w))
                processed_batch.append(processed_ep)
        
        # Post-process each episode in the batch
        for eps in processed_batch:
            # Single-pass computation: generate joint_action and delta_joint_pos in one go
            if 'robot_joint' in eps and len(eps['robot_joint']) > 1:
                robot_joints = eps['robot_joint']
                
                # Use next timestep's robot_joint as joint_action (target positions)
                eps['joint_action'] = robot_joints[1:]  # Next timestep's positions
                
                # Compute delta_joint_pos using next timestamp approach
                delta_joint_pos = robot_joints[1:] - robot_joints[:-1]  # More efficient
                eps['delta_joint_pos'] = delta_joint_pos

                # Truncate all arrays to match delta length (length - 1)
                final_length = len(robot_joints) - 1
                for key in list(eps.keys()):
                    if hasattr(eps[key], '__len__') and len(eps[key]) > final_length:
                        eps[key] = eps[key][:final_length]

            # Optimized dtype conversion
            for k, v in eps.items():
                if hasattr(v, 'dtype') and v.dtype == np.float64:   # float64 -> float32
                    eps[k] = v.astype(np.float32, copy=False)  # In-place conversion when possible

            # Add to new buffer (deferred compression)
            new_buffer.add_episode(eps)
        
        # Progress tracking and ETA calculation
        batch_time = time.time() - batch_start_time
        elapsed_total = time.time() - start_time
        avg_batch_time = elapsed_total / (batch_idx + 1)
        remaining_batches = total_batches - (batch_idx + 1)
        eta_seconds = remaining_batches * avg_batch_time
        
        episodes_processed = batch_end
        processing_rate = episodes_processed / elapsed_total
        
        print(f"✅ Batch {batch_idx+1}/{total_batches} completed: episodes {batch_start+1}-{batch_end}/{total_episodes}")
        print(f"   ⏱️  Batch time: {batch_time:.1f}s | Total elapsed: {elapsed_total/60:.1f}m | ETA: {eta_seconds/60:.1f}m")
        print(f"   📊 Processing rate: {processing_rate:.1f} episodes/second")
        print()
    
    print(f"✅ Processing complete! Added {new_buffer.n_episodes} episodes to new buffer")
    assert new_buffer.n_episodes == old_buffer.n_episodes

    # Apply compression only at final save for better performance
    print("Saving with compression (this may take a moment)...")
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    new_buffer.save_to_path(save_path, compressors=compressor)

    try:
        test_buffer = ReplayBuffer.create_from_path(save_path)
        print(f"Original buffer: \n{old_buffer}")
        print(f"Successfully saved to {save_path}, check the buffer: \n{test_buffer}")
        
        # Generate videos if requested
        if generate_video:
            generate_inspection_videos(test_buffer, save_path)
            
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        print(f"Full traceback:")
        traceback.print_exc()
        raise


def generate_inspection_videos(buffer: ReplayBuffer, save_path: str):
    """Generate videos from the converted dataset for manual inspection"""
    video_dir = Path(save_path).parent / f"{Path(save_path).stem}_videos"
    video_dir.mkdir(exist_ok=True)
    
    print(f"Generating inspection videos in {video_dir}")
    
    # Check available camera streams
    camera_keys = [k for k in buffer.keys if 'camera' in k and 'color' in k]
    
    for episode_idx in range(min(5, buffer.n_episodes)):  # Generate videos for first 5 episodes
        episode = buffer.get_episode(episode_idx)
        
        for camera_key in camera_keys:
            if camera_key not in episode:
                continue
                
            # Get images - handle different data formats
            images = episode[camera_key]
            if len(images.shape) == 4 and images.shape[1] == 3:  # (T, 3, H, W)
                images = np.transpose(images, (0, 2, 3, 1))  # Convert to (T, H, W, 3)
            
            # Ensure uint8 format
            if images.dtype != np.uint8:
                images = (images * 255).astype(np.uint8) if images.max() <= 1.0 else images.astype(np.uint8)
            
            # Create video
            video_path = video_dir / f"episode_{episode_idx}_{camera_key}.mp4"
            height, width = images.shape[1:3]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))
            
            for img in images:
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)
            
            out.release()
            print(f"Generated video: {video_path}")
    
    print(f"Video generation complete. Check {video_dir} for inspection videos.")


if __name__ == '__main__':
    main()

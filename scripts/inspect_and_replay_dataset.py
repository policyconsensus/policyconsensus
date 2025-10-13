#!/usr/bin/env python3
"""
Unified script to inspect and replay dataset episodes from zarr files.
Combines dataset structure inspection with video generation capabilities.
Supports various dataset formats (RLBench, real-world tasks, etc.)

Usage: 
  # Inspect only
  python scripts/inspect_and_replay_dataset.py <zarr_path> --inspect-only
  
  # Generate videos only  
  python scripts/inspect_and_replay_dataset.py <zarr_path> --video-only
  
  # Both inspection and video generation (default)
  python scripts/inspect_and_replay_dataset.py <zarr_path>
"""

import os
import sys
import pathlib
import argparse
import numpy as np
import zarr
import cv2
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from scipy.ndimage import gaussian_filter

# Add project root to path
ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from modular_policy.common.replay_buffer import ReplayBuffer
from modular_policy.workspace.eval_visualization import get_tactile_viz


def analyze_zarr_structure(zarr_path: str) -> Optional[zarr.Group]:
    """Analyze the structure of a zarr dataset."""
    print(f"🔍 Analyzing zarr dataset: {zarr_path}")
    
    original_zarr_path = zarr_path # Store original path for error messages

    try:
        # First, try to open the path directly
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Dataset not found at {zarr_path}")
        zarr_file = zarr.open(zarr_path, mode='r')
        print(f"✅ Successfully opened zarr dataset")
        print(f"📁 Top-level groups: {list(zarr_file.keys())}")
        
    except Exception as e:
        # If direct open fails, and it's a directory, try appending replay_buffer.zarr
        if os.path.isdir(original_zarr_path):
            potential_zarr_path = os.path.join(original_zarr_path, "replay_buffer.zarr")
            print(f"⚠️  Could not open '{original_zarr_path}' directly ({e}). Trying: {potential_zarr_path}")
            if os.path.exists(potential_zarr_path):
                try:
                    zarr_file = zarr.open(potential_zarr_path, mode='r')
                    print(f"✅ Successfully opened zarr dataset at {potential_zarr_path}")
                    print(f"📁 Top-level groups: {list(zarr_file.keys())}")
                    # Update zarr_path to the resolved path for subsequent calls
                    zarr_path = potential_zarr_path
                except Exception as inner_e:
                    print(f"❌ Error analyzing zarr file at {potential_zarr_path}: {inner_e}")
                    return None
            else:
                print(f"❌ Directory found at {original_zarr_path}, but replay_buffer.zarr not found inside.")
                return None
        else:
            print(f"❌ Error analyzing zarr file: {e}")
            return None
        
    # Check data structure (moved outside try-except for clarity)
    if 'data' in zarr_file:
        data_group = zarr_file['data']
        print(f"📊 Data keys: {list(data_group.keys())}")
        
        # Get basic info about each data type
        for key in data_group.keys():
            try:
                data_array = data_group[key]
                if hasattr(data_array, 'shape'):
                    size_mb = data_array.nbytes / (1024*1024)
                    print(f"  {key}: shape={data_array.shape}, dtype={data_array.dtype} [{size_mb:.1f}MB]")
                else:
                    print(f"  {key}: Group with {len(data_array.keys())} items")
                    pass # No print for groups
            except Exception as e:
                print(f"  {key}: Error inspecting - {e}")
                pass # No print for errors
            
    # Check meta information
    if 'meta' in zarr_file:
        meta_group = zarr_file['meta']
        print(f"📋 Meta keys: {list(meta_group.keys())}")
        
        # Check episode ends if available
        if 'episode_ends' in meta_group:
            episode_ends = meta_group['episode_ends'][:]
            print(f"  Number of episodes: {len(episode_ends)}")
            print(f"  Total timesteps: {episode_ends[-1] if len(episode_ends) > 0 else 0}")
            if len(episode_ends) > 0:
                episode_lengths = np.diff([0] + episode_ends[:5].tolist())
                print(f"  First few episode lengths: {episode_lengths}")
            
    return zarr_file


def inspect_dataset_detailed(zarr_path: str, show_samples: bool = True, max_episodes: int = 3) -> None:
    """Perform detailed dataset inspection with quality checks."""
    print(f"\n{'='*80}")
    print(f"🔍 DETAILED DATASET INSPECTION")
    print(f"{'='*80}")
    
    try:
        # First, try direct zarr access for basic info (faster)
        zarr_root = zarr.open(zarr_path, mode='r')
        
        # Access data group
        if 'data' in zarr_root:
            data_group = zarr_root['data']
            meta_group = zarr_root.get('meta', None)
            
            # Check if this is a nested structure (raw data)
            has_nested_structure = any(not hasattr(data_group[k], 'shape') for k in data_group.keys() 
                                     if k not in ['stage', 'timestamp'])
            
            if has_nested_structure:
                print("⚠️  Detected nested data structure (raw format)")
                print("    Use --no-samples for basic inspection of nested datasets")
                if show_samples:
                    print("    Detailed sampling not supported for nested structures")
                    show_samples = False
            
            # Basic statistics from zarr (no memory loading)
            print(f"\n📊 Dataset Statistics:")
            
            # Get episode info from meta if available
            if meta_group and 'episode_ends' in meta_group:
                episode_ends = np.array(meta_group['episode_ends'])
                n_episodes = len(episode_ends)
                n_steps = episode_ends[-1] if len(episode_ends) > 0 else 0
                
                print(f"  Number of episodes: {n_episodes}")
                print(f"  Total steps: {n_steps}")
                
                if n_episodes > 0:
                    episode_lengths = np.diff(np.concatenate([[0], episode_ends]))
                    print(f"  Average episode length: {np.mean(episode_lengths):.1f} steps")
                    print(f"  Min episode length: {np.min(episode_lengths)} steps")
                    print(f"  Max episode length: {np.max(episode_lengths)} steps")
            else:
                # Fallback: estimate from first data array
                first_key = list(data_group.keys())[0]
                total_steps = data_group[first_key].shape[0] if hasattr(data_group[first_key], 'shape') else 0
                print(f"  Total steps: {total_steps}")
                print(f"  Episode info: Not available in meta")
        
        # Inspect observation keys (non-action keys)
        all_keys = list(data_group.keys())
        obs_keys = [k for k in all_keys if not any(x in k.lower() for x in ['action', 'reward', 'terminal'])]
        action_keys = [k for k in all_keys if 'action' in k.lower()]
        other_keys = [k for k in all_keys if any(x in k.lower() for x in ['reward', 'terminal'])]
        
        print(f"\n🎯 Data categorization:")
        print(f"  Observation keys: {obs_keys}")
        print(f"  Action keys: {action_keys}")
        print(f"  Other keys: {other_keys}")
            
        # Fast quality checks using zarr data
        print(f"\n🔍 Data Quality Checks:")
        
        def check_data_quality(group, prefix="", sample_size=1000):
            results = {"zero_var": [], "nan_values": [], "image_format": []}
            
            for key in group.keys():
                full_key = f"{prefix}{key}" if prefix else key
                item = group[key]
                
                try:
                    if hasattr(item, 'shape') and len(item.shape) > 0:
                        # Sample data to check variance
                        sample_end = min(sample_size, item.shape[0])
                        if item.dtype in [np.float32, np.float64]:
                            sample_data = np.array(item[:sample_end])
                            
                            # Check for NaN
                            if np.isnan(sample_data).any():
                                results["nan_values"].append(full_key)
                            
                            # Check for zero variance
                            if sample_data.size > 1 and np.var(sample_data) == 0:
                                results["zero_var"].append(full_key)
                        
                        # Check image format
                        if 'color' in key.lower() or key.endswith('_rgb'):
                            results["image_format"].append((full_key, item.shape, item.dtype))
                    
                    elif hasattr(item, 'keys'):
                        # Recurse into nested groups
                        nested_results = check_data_quality(item, f"{full_key}/")
                        for result_type in results:
                            results[result_type].extend(nested_results[result_type])
                
                except Exception as e:
                    print(f"    ⚠️  Error checking {full_key}: {e}")
                    pass # No print for errors
            
            return results
        
        # Run quality checks
        quality_results = check_data_quality(data_group)
        
        # Report results
        if quality_results["zero_var"]:
            print(f"  ⚠️  Zero variance found in: {quality_results['zero_var']}")
            pass # No print for zero variance
        else:
            print(f"  ✅ All numerical data has variance")
            pass # No print for variance
        
        if quality_results["nan_values"]:
            print(f"  ⚠️  NaN values found in: {quality_results['nan_values']}")
            pass # No print for NaN values
        else:
            print(f"  ✅ No NaN values found")
            pass # No print for no NaN values
        
        # Check image data format
        if quality_results["image_format"]:
            print(f"  📷 Found {len(quality_results['image_format'])} RGB observation(s):")
            for img_key, shape, dtype in quality_results["image_format"]:
                print(f"    {img_key}: shape={shape}, dtype={dtype}")
                if dtype == np.uint8:
                    print(f"      ✅ Correct uint8 format for images")
                    pass # No print for correct format
                else:
                    print(f"      ⚠️  Expected uint8, got {dtype}")
                    pass # No print for incorrect format
        
        # Only load ReplayBuffer for detailed sampling if requested
        if show_samples:
            print(f"\n📖 Loading ReplayBuffer for sample inspection...")
            try:
                # Use the already opened zarr_root to create ReplayBuffer without copying to memory
                replay_buffer = ReplayBuffer(root=zarr_root)
                print(f"✅ ReplayBuffer loaded successfully (using direct zarr access)")
                
                if replay_buffer.n_episodes > 0:
                    print(f"\n📖 Sample Episodes (showing first {min(max_episodes, replay_buffer.n_episodes)}):")
                    
                    for ep_idx in range(min(max_episodes, replay_buffer.n_episodes)):
                        print(f"\n  Episode {ep_idx}:")
                        
                        # Get episode data
                        start_idx = replay_buffer.episode_ends[ep_idx-1] if ep_idx > 0 else 0
                        end_idx = replay_buffer.episode_ends[ep_idx]
                        episode_length = end_idx - start_idx
                        
                        print(f"    Length: {episode_length} steps")
                        print(f"    Steps: {start_idx} to {end_idx-1}")
                        
                        # Sample some data (limited keys to avoid memory issues)
                        sample_keys = list(replay_buffer.keys())[:5]
                        for key in sample_keys:
                            data = replay_buffer[key][start_idx:min(start_idx+3, end_idx)]
                            
                            if 'color' in key or key.endswith('_rgb'):
                                print(f"    {key}: {data.shape} (RGB image)")
                                if len(data) > 0:
                                    print(f"      Value range: [{data.min()}, {data.max()}]")
                                    pass # No print for value range
                            elif 'action' in key:
                                print(f"    {key}: {data.shape}")
                                if len(data) > 0:
                                    print(f"      Value range: [{data.min():.3f}, {data.max():.3f}]")
                                    print(f"      First action: {data[0]}")
                                    pass # No print for action values
                            else:
                                print(f"    {key}: {data.shape}")
                                if len(data) > 0 and data.dtype in [np.float32, np.float64]:
                                    print(f"      Value range: [{data.min():.3f}, {data.max():.3f}]")
                                    pass # No print for value range
                    
            except Exception as replay_error:
                print(f"  ⚠️  Could not load ReplayBuffer for detailed sampling: {replay_error}")
                print(f"  📖 Skipping detailed sampling")
                pass # No print for replay buffer errors
        
    except Exception as e:
        print(f"❌ Error during detailed inspection: {e}")
        pass # No print for detailed inspection errors


def get_episode_slice(zarr_file: zarr.Group, episode_idx: int) -> Tuple[int, int]:
    """Get the start and end indices for a specific episode."""
    try:
        meta_group = zarr_file['meta']
        episode_ends = meta_group['episode_ends'][:]
        
        start_idx = 0 if episode_idx == 0 else episode_ends[episode_idx - 1]
        end_idx = episode_ends[episode_idx]
        
        return start_idx, end_idx
    except Exception as e:
        print(f"ERROR getting episode slice: {e}")
        return 0, 0


def get_image_keys(data_group: zarr.Group, prefix: str = "") -> Tuple[List[str], List[str]]:
    """Recursively get all available camera and tactile data keys from the dataset."""
    camera_keys = []
    tactile_keys = []
    
    # Common camera key patterns to look for
    common_camera_patterns = [
        'agentview_rgb', 'robot0_eye_in_hand_rgb', 'frontview_rgb',
        'wrist_rgb', 'overhead_rgb', 'image', 'rgb', 'camera_rgb',
        'left_shoulder_rgb', 'right_shoulder_rgb', 'wrist_camera_rgb'
    ]
    
    for key in data_group.keys():
        full_key = f"{prefix}{key}" if prefix else key
        item = data_group[key]

        if isinstance(item, zarr.Group):
            # If it's a nested group, recurse
            nested_camera_keys, nested_tactile_keys = get_image_keys(item, prefix=f"{full_key}/")
            camera_keys.extend(nested_camera_keys)
            tactile_keys.extend(nested_tactile_keys)
        elif hasattr(item, 'shape'):
            # Check for tactile data
            if 'tactile' in key.lower():
                tactile_keys.append(full_key)
            # Check for camera data
            elif any(pattern in key.lower() for pattern in common_camera_patterns):
                camera_keys.append(full_key)
            elif (any(img_key in key.lower() for img_key in ['rgb', 'image', 'camera']) and
                  not any(exclude in key.lower() for exclude in ['depth', 'intrinsic', 'extrinsic']) and
                  key.endswith('_color')):
                camera_keys.append(full_key)
            elif 'rgb' in key.lower() and not any(exclude in key.lower() for exclude in ['depth', 'intrinsic', 'extrinsic']):
                camera_keys.append(full_key)
    
    return camera_keys, tactile_keys


def apply_gaussian_blur(contact_map, sigma=0.1):
    return gaussian_filter(contact_map, sigma=sigma)

def temporal_filter(new_frame, prev_frame, alpha=0.2):
    """
    Apply temporal smoothing filter.
    'alpha' determines the blending factor.
    A higher alpha gives more weight to the current frame, while a lower alpha gives more weight to the previous frame.
    """
    return alpha * new_frame + (1 - alpha) * prev_frame




def process_image_frame(img: np.ndarray, is_tactile: bool = False, prev_tactile_frame: Optional[np.ndarray] = None) -> np.ndarray:
    """Process a single image frame to prepare it for video writing."""
    if is_tactile:
        print(f"DEBUG: Processing tactile frame - original shape: {img.shape}, dtype: {img.dtype}, range: [{img.min():.3f}, {img.max():.3f}]")
        
        # Handle tactile data structure matching demo_real_ur5e.py approach
        # Expected: (1, 16, 32) -> should become (16, 32)
        if len(img.shape) == 3:
            if img.shape[0] == 1 and img.shape[1] == 16 and img.shape[2] == 32:
                # This is tactile data with shape (1, 16, 32) -> extract (16, 32)
                img = img[0]  # Remove batch dimension: (1, 16, 32) -> (16, 32)
                print(f"DEBUG: Extracted tactile grid from (1, 16, 32) -> {img.shape}")
            elif img.shape[0] == 1:
                # General case: (1, H, W) -> (H, W)
                img = img[0]
                print(f"DEBUG: Removed batch dimension: -> {img.shape}")
            elif img.shape[-1] == 3 or img.shape[-1] == 1:
                # Traditional (H, W, C) image format
                if img.shape[-1] > 1:
                    img = np.mean(img, axis=-1) # Result: (H, W)
                else: # If it's (H, W, 1)
                    img = img.squeeze(-1) # Result: (H, W)
            else:
                # Unexpected 3D shape, try to handle gracefully
                print(f"WARNING: Unexpected 3D tactile shape {img.shape}, taking first slice")
                img = img[0]
        elif len(img.shape) == 1: # If it's a flattened 1D array, try to reshape
            print(f"DEBUG: 1D tactile data with {img.size} elements")
            
            # Based on tactile sensor structure: 16 rows × 32 columns = 512 elements
            if img.size == 512:
                # Reshape to 16 (rows) × 32 (columns) grid - tactile sensor layout
                img = img.reshape(16, 32)
                print(f"DEBUG: Reshaped 1D tactile data to 16×32 grid (rows×columns)")
            elif img.size == 16*16:
                img = img.reshape(16, 16)
                print(f"DEBUG: Reshaped to 16×16 grid")
            elif img.size == 32*32:
                img = img.reshape(32, 32)
                print(f"DEBUG: Reshaped to 32×32 grid")
            elif img.size == 64*64:
                img = img.reshape(64, 64)
                print(f"DEBUG: Reshaped to 64×64 grid")
            else:
                print(f"DEBUG: Non-standard tactile size: {img.size} elements")
                # Try to find reasonable height/width factors
                if img.size % 16 == 0:
                    width = img.size // 16
                    img = img.reshape(16, width)
                    print(f"DEBUG: Reshaped to 16×{width} grid")
                elif img.size % 32 == 0:
                    height = img.size // 32
                    img = img.reshape(height, 32)
                    print(f"DEBUG: Reshaped to {height}×32 grid")
                else:
                    # Find best rectangular shape
                    side_len = int(np.sqrt(img.size))
                    if side_len * side_len == img.size:
                        img = img.reshape(side_len, side_len)
                        print(f"DEBUG: Reshaped to {side_len}×{side_len} square grid")
                    else:
                        print(f"ERROR: Cannot find suitable grid shape for {img.size} elements")
                        return np.zeros((64, 64, 3), dtype=np.uint8)
        elif len(img.shape) != 2:
            print(f"WARNING: Unexpected tactile image shape: {img.shape}. Attempting to force to 2D.")
            # Try to flatten to 2D if it's more than 2D and not (H,W,C)
            if len(img.shape) > 2:
                img = np.mean(img, axis=tuple(range(2, len(img.shape))))  # Mean over all dims after first 2
            else:
                img = img.reshape(img.shape[0], -1) # Flatten remaining dimensions

        print(f"DEBUG: After reshaping - shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")

        # ============================================================================
        # Use the working approach from demo_real_ur5e.py:63-68
        # Simple scaling to 0-255 (assumes data is already properly normalized 0-1)
        # ============================================================================
        
        # Scale to 0-255 (demo_real_ur5e.py approach)
        img_scaled = (img * 255).astype(np.uint8)
        print(f"DEBUG: After scaling to 0-255 - shape: {img_scaled.shape}, dtype: {img_scaled.dtype}, range: [{img_scaled.min()}, {img_scaled.max()}]")
        # Apply VIRIDIS colormap (demo_real_ur5e.py approach)
        try:
            img_colored = cv2.applyColorMap(img_scaled, cv2.COLORMAP_VIRIDIS)
            print(f"DEBUG: Applied VIRIDIS colormap - shape: {img_colored.shape}")
        except Exception as e:
            print(f"WARNING: Could not apply colormap, using grayscale: {e}")
            # Fallback to grayscale to BGR conversion
            if len(img_scaled.shape) == 2:
                img_colored = cv2.cvtColor(img_scaled, cv2.COLOR_GRAY2BGR)
            else:
                print(f"ERROR: Cannot convert to BGR: {img_scaled.shape}")
                debug_img = np.zeros((64, 64, 3), dtype=np.uint8)
                debug_img[:, :, 0] = 128  # Red channel for debugging
                return debug_img
        
        print(f"DEBUG: Final tactile frame - shape: {img_colored.shape}, range: [{img_colored.min()}, {img_colored.max()}]")
        return img_colored

    else:
        # Original camera image processing
        if img.shape[0] == 3 and len(img.shape) == 3:  # CHW format
            img = img.transpose(1, 2, 0)  # Convert to HWC
        
        # if img.dtype == np.float32 or img.dtype == np.float64:
        #     img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    return img


def create_episode_video(
    zarr_file: zarr.Group, 
    episode_idx: int, 
    output_dir: str = "output/replay_videos", 
    fps: int = 10,
    dataset_name: str = "dataset",
    touch_only: bool = False
) -> bool:
    """Create a video from episode observations."""
    print(f"🎬 Creating video for episode {episode_idx}...")
    
    try:
        data_group = zarr_file['data']
        
        # Get episode boundaries
        start_idx, end_idx = get_episode_slice(zarr_file, episode_idx)
        if end_idx <= start_idx:
            print(f"❌ Invalid episode indices {start_idx}-{end_idx}")
            return False
            
        print(f"Episode {episode_idx}: timesteps {start_idx} to {end_idx-1} (length: {end_idx - start_idx})")
        
        # Extract episode data slice
        episode_slice = slice(start_idx, end_idx)
        
        # Get all available camera and tactile keys
        camera_keys, tactile_keys = get_image_keys(data_group, prefix="")
        
        # Filter based on touch_only mode
        if touch_only:
            camera_keys = []  # Ignore camera keys in touch-only mode
            if not tactile_keys:
                print("❌ No tactile data found in dataset for touch-only video")
                return False
            print(f"✋ Found tactile keys (touch-only mode): {tactile_keys}")
        else:
            if not camera_keys and not tactile_keys:
                print("❌ No camera or tactile data found in dataset")
                return False
            print(f"📷 Found camera keys: {camera_keys}")
            print(f"✋ Found tactile keys: {tactile_keys}")
        
        # Load camera data for the episode
        camera_obs_data = {}
        for key in camera_keys:
            if key in data_group:
                camera_obs_data[key] = data_group[key][episode_slice]
                print(f"  {key}: {camera_obs_data[key].shape}")

        # Load tactile data for the episode
        tactile_obs_data = {}
        for key in tactile_keys:
            if key in data_group:
                tactile_obs_data[key] = data_group[key][episode_slice]
                print(f"  {key}: {tactile_obs_data[key].shape}")
        
        # Get actions if available
        if 'action' in data_group:
            actions = data_group['action'][episode_slice]
            print(f"  actions: {actions.shape}")
        
        # Create output directory
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create video
        return _write_video_file(camera_obs_data, tactile_obs_data, episode_idx, output_dir, fps, dataset_name, touch_only)
        
    except Exception as e:
        print(f"❌ Error creating episode video: {e}")
        import traceback
        traceback.print_exc() # Keep traceback for debugging errors, but not verbose output
        return False


def _write_video_file(
    camera_obs_data: Dict[str, np.ndarray],
    tactile_obs_data: Dict[str, np.ndarray],
    episode_idx: int,
    output_dir: str,
    fps: int,
    dataset_name: str,
    touch_only: bool = False
) -> bool:
    """Write the actual video file from observation data."""
    video_suffix = "_touch_only" if touch_only else ""
    video_path = pathlib.Path(output_dir) / f"{dataset_name}_episode_{episode_idx}{video_suffix}.mp4"
    
    num_camera_frames = len(list(camera_obs_data.values())[0]) if camera_obs_data else 0
    num_tactile_frames = len(list(tactile_obs_data.values())[0]) if tactile_obs_data else 0

    if num_camera_frames == 0 and num_tactile_frames == 0:
        print("❌ No frames to write (camera or tactile)")
        return False

    num_frames = max(num_camera_frames, num_tactile_frames)
    
    # Determine frame dimensions from camera data, or tactile if no camera
    frame_height, frame_width = 0, 0
    if touch_only and tactile_obs_data:
        # For touch-only mode, create a sample tactile visualization to get dimensions
        # Prepare sample obs in the format expected by get_tactile_viz (with time dimension)
        sample_obs = {}
        tactile_keys_list = list(tactile_obs_data.keys())
        
        # Map tactile keys to left/right format expected by get_tactile_viz
        # get_tactile_viz expects obs[key][-1][0] to give (16, 32) array
        for i, key in enumerate(tactile_keys_list):
            # Get first sample and reshape from (1, 16, 32) to (16, 32) by removing batch dimension
            sample_data = list(tactile_obs_data.values())[i][0:1]  # Shape: (1, 1, 16, 32)
            # Reshape to format expected by get_tactile_viz: list with (1, 16, 32) arrays
            formatted_data = [sample_data[0]]  # Remove time dimension: (1, 16, 32)
            
            if 'left' in key.lower():
                sample_obs['left_tactile'] = formatted_data
            elif 'right' in key.lower():
                sample_obs['right_tactile'] = formatted_data
            else:
                # If we can't determine left/right, use first key as left, second as right
                if 'left_tactile' not in sample_obs:
                    sample_obs['left_tactile'] = formatted_data
                elif 'right_tactile' not in sample_obs:
                    sample_obs['right_tactile'] = formatted_data
        
        # Handle cases where we might only have one tactile sensor
        if 'left_tactile' not in sample_obs and 'right_tactile' in sample_obs:
            sample_obs['left_tactile'] = [np.zeros_like(sample_obs['right_tactile'][0])]
        elif 'right_tactile' not in sample_obs and 'left_tactile' in sample_obs:
            sample_obs['right_tactile'] = [np.zeros_like(sample_obs['left_tactile'][0])]
                
        touch_viz = get_tactile_viz(sample_obs)
        if touch_viz is not None:
            frame_height, frame_width = touch_viz.shape[:2]
    elif camera_obs_data:
        first_camera_frame = process_image_frame(list(camera_obs_data.values())[0][0])
        frame_height, frame_width = first_camera_frame.shape[:2]
    elif tactile_obs_data:
        # Process a tactile frame to get its display dimensions
        first_tactile_frame = process_image_frame(list(tactile_obs_data.values())[0][0], is_tactile=True)
        frame_height, frame_width = first_tactile_frame.shape[:2]

    if frame_height == 0 or frame_width == 0:
        print("❌ Could not determine frame dimensions.")
        return False

    # Determine video layout
    num_cameras = len(camera_obs_data)
    num_tactiles = len(tactile_obs_data)

    # Calculate total width and height for the combined frame
    # We'll arrange cameras in a grid, and tactile sensors below them
    
    # Camera grid dimensions
    camera_cols = min(num_cameras, 3) if num_cameras > 0 else 0
    camera_rows = (num_cameras + camera_cols - 1) // camera_cols if camera_cols > 0 else 0
    
    # Tactile grid dimensions (assuming 2 tactile sensors, left and right)
    tactile_cols = min(num_tactiles, 2) if num_tactiles > 0 else 0
    tactile_rows = (num_tactiles + tactile_cols - 1) // tactile_cols if tactile_cols > 0 else 0

    # Calculate overall video dimensions
    if touch_only and tactile_obs_data:
        # For touch-only mode, use the actual dimensions of get_tactile_viz() output
        video_width = frame_width   # 800
        video_height = frame_height # 800
    else:
        # Max width is determined by the wider of camera or tactile rows
        video_width = max(camera_cols * frame_width, tactile_cols * frame_width)
        video_height = (camera_rows * frame_height) + (tactile_rows * frame_height)

        # Ensure minimum width if no cameras or tactiles, but frames exist
        if video_width == 0 and (num_camera_frames > 0 or num_tactile_frames > 0):
            video_width = frame_width
        if video_height == 0 and (num_camera_frames > 0 or num_tactile_frames > 0):
            video_height = frame_height

    if video_width == 0 or video_height == 0:
        print("❌ Calculated video dimensions are zero. Cannot create video.")
        return False

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (video_width, video_height))
    
    # Initialize previous tactile frame for temporal filtering
    prev_tactile_frame = None

    try:
        camera_keys = list(camera_obs_data.keys())
        tactile_keys = list(tactile_obs_data.keys())
        
        for frame_idx in range(num_frames):
            combined_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            current_y_offset = 0

            # Handle touch-only mode
            if touch_only and tactile_obs_data:
                # Create touch-only visualization using existing get_tactile_viz function
                # Format the data as expected by get_tactile_viz (with time dimension)
                touch_obs = {}
                tactile_keys_list = list(tactile_obs_data.keys())
                
                # Try to map tactile keys to left/right format expected by get_tactile_viz
                # get_tactile_viz expects obs[key][-1][0] to give (16, 32) array
                for key in tactile_keys_list:
                    # Get current frame data using the actual key: shape (1, 16, 32)
                    frame_data = tactile_obs_data[key][frame_idx]  # Shape: (1, 16, 32)
                    # Format as list for get_tactile_viz: expects list of (1, 16, 32) arrays
                    formatted_data = [frame_data]
                    
                    if 'left' in key.lower():
                        touch_obs['left_tactile'] = formatted_data
                    elif 'right' in key.lower():
                        touch_obs['right_tactile'] = formatted_data
                    else:
                        # If we can't determine left/right, use first key as left, second as right
                        if 'left_tactile' not in touch_obs:
                            touch_obs['left_tactile'] = formatted_data
                        elif 'right_tactile' not in touch_obs:
                            touch_obs['right_tactile'] = formatted_data
                
                # Fill in missing tactile data with zeros if needed
                if 'left_tactile' not in touch_obs and 'right_tactile' in touch_obs:
                    touch_obs['left_tactile'] = [np.zeros_like(touch_obs['right_tactile'][0])]
                elif 'right_tactile' not in touch_obs and 'left_tactile' in touch_obs:
                    touch_obs['right_tactile'] = [np.zeros_like(touch_obs['left_tactile'][0])]
                
                touch_viz = get_tactile_viz(touch_obs)
                if touch_viz is not None:
                    # For touch-only mode, don't resize - the tactile viz should already be the right size
                    combined_frame = touch_viz
                
            # Render camera frames (skip in touch-only mode)
            elif num_cameras > 0:
                camera_grid_height = camera_rows * frame_height
                camera_grid_width = camera_cols * frame_width
                
                # Create a temporary canvas for camera grid to handle potential width mismatch
                camera_canvas = np.zeros((camera_grid_height, camera_grid_width, 3), dtype=np.uint8)

                for i, key in enumerate(camera_keys):
                    row = i // camera_cols
                    col = i % camera_cols
                    
                    img = process_image_frame(camera_obs_data[key][frame_idx])
                    
                    # Resize to fit grid cell
                    if img.shape[:2] != (frame_height, frame_width):
                        img = cv2.resize(img, (frame_width, frame_height))
                    
                    # Place in grid
                    y_start = row * frame_height
                    y_end = y_start + frame_height
                    x_start = col * frame_width
                    x_end = x_start + frame_width
                    
                    camera_canvas[y_start:y_end, x_start:x_end] = img
                    
                    # Add camera label
                    cv2.putText(camera_canvas, key, (x_start + 10, y_start + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Copy camera canvas to combined frame, centering if necessary
                x_offset = (video_width - camera_canvas.shape[1]) // 2
                combined_frame[current_y_offset:current_y_offset + camera_grid_height,
                               x_offset:x_offset + camera_canvas.shape[1]] = camera_canvas
                current_y_offset += camera_grid_height

            # Render tactile frames (skip in touch-only mode)
            if num_tactiles > 0 and not touch_only:
                tactile_grid_height = tactile_rows * frame_height
                tactile_grid_width = tactile_cols * frame_width

                # Create a temporary canvas for tactile grid
                tactile_canvas = np.zeros((tactile_grid_height, tactile_grid_width, 3), dtype=np.uint8)

                for i, key in enumerate(tactile_keys):
                    row = i // tactile_cols
                    col = i % tactile_cols
                    
                    # Pass prev_tactile_frame to process_image_frame
                    current_tactile_data = tactile_obs_data[key][frame_idx]
                    img = process_image_frame(current_tactile_data, is_tactile=True, prev_tactile_frame=prev_tactile_frame)
                    
                    # Update prev_tactile_frame for the next iteration
                    # Ensure prev_tactile_frame is 2D for temporal filter
                    if len(current_tactile_data.shape) == 1 and current_tactile_data.size == 512:
                        prev_tactile_frame = current_tactile_data.reshape(16, 32)
                    else:
                        prev_tactile_frame = current_tactile_data # Assuming it's already 2D or will be handled by process_image_frame

                    # Resize to fit grid cell
                    if img.shape[:2] != (frame_height, frame_width):
                        img = cv2.resize(img, (frame_width, frame_height))
                    
                    # Place in grid
                    y_start = row * frame_height
                    y_end = y_start + frame_height
                    x_start = col * frame_width
                    x_end = x_start + frame_width
                    
                    tactile_canvas[y_start:y_end, x_start:x_end] = img
                    
                    # Add tactile label
                    cv2.putText(tactile_canvas, key, (x_start + 10, y_start + 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Copy tactile canvas to combined frame, centering if necessary
                x_offset = (video_width - tactile_canvas.shape[1]) // 2
                combined_frame[current_y_offset:current_y_offset + tactile_grid_height,
                               x_offset:x_offset + tactile_canvas.shape[1]] = tactile_canvas
                current_y_offset += tactile_grid_height

            # Frame counter removed per user request
            
            video_writer.write(combined_frame)
        
        video_writer.release()
        print(f"✅ Video saved to: {video_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error writing video: {e}")
        video_writer.release()
        # Remove incomplete video file
        if video_path.exists():
            video_path.unlink()
        return False


def replay_dataset_episodes(
    zarr_path: str,
    episode_indices: Optional[List[int]] = None,
    output_dir: str = "output/replay_videos",
    fps: int = 10,
    max_episodes: int = 100,
    touch_only: bool = False
) -> None:
    """Generate videos from multiple episodes in a dataset."""
    
    print(f"\n{'='*80}")
    print(f"🎬 VIDEO GENERATION")
    print(f"{'='*80}")
    
    # Open dataset
    zarr_file = zarr.open(zarr_path, mode='r')
    if zarr_file is None:
#        print("❌ Could not open dataset for video generation")
        return
    
    # Get dataset name from path
    dataset_name = pathlib.Path(zarr_path).stem
    
    # Determine which episodes to replay
    if episode_indices is None:
        # Default: replay first few episodes
        try:
            meta_group = zarr_file['meta']
            episode_ends = meta_group['episode_ends'][:]
            total_episodes = len(episode_ends)
            episode_indices = list(range(min(max_episodes, total_episodes)))
        except:
            print("❌ Could not determine episode count, using episode 0")
            episode_indices = [0]
    
    print(f"🎬 Generating videos for {len(episode_indices)} episodes...")
    
    # Create videos for each episode
    success_count = 0
    for ep_idx in episode_indices:
#        print(f"\n=== Episode {ep_idx} ===")
        if create_episode_video(zarr_file, ep_idx, output_dir, fps, dataset_name, touch_only):
            success_count += 1
    
    print(f"\n✅ Video generation completed! Successfully created {success_count}/{len(episode_indices)} videos.")
    print(f"📁 Videos saved to: {output_dir}")


def main():
    """Main function for unified dataset inspection and replay."""
    parser = argparse.ArgumentParser(
        description="Unified dataset inspection and video replay tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full inspection + video generation (default)
  python scripts/inspect_and_replay_dataset.py data/rlbench/rlbench_mt4_expert_100.zarr

  # Inspection only
  python scripts/inspect_and_replay_dataset.py data/rlbench/rlbench_mt4_expert_100.zarr --inspect-only

  # Video generation only
  python scripts/inspect_and_replay_dataset.py data/rlbench/rlbench_mt4_expert_100.zarr --video-only

  # Custom episodes for videos
  python scripts/inspect_and_replay_dataset.py data/rlbench/rlbench_mt4_expert_100.zarr --episodes 0 1 5
  
  # Touch-only video generation
  python scripts/inspect_and_replay_dataset.py data/0823_marker_expert_10.zarr --touch-only
        """
    )
    
    parser.add_argument("zarr_path", help="Path to zarr dataset")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--inspect-only", action="store_true", 
                           help="Only inspect dataset structure and quality")
    mode_group.add_argument("--video-only", action="store_true",
                           help="Only generate videos, skip inspection")
    
    # Inspection options
    parser.add_argument("--no-samples", action="store_true", 
                       help="Skip detailed sample inspection (faster)")
    parser.add_argument("--max-inspect-episodes", type=int, default=3,
                       help="Max episodes to inspect in detail (default: 3)")
    
    # Video options
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                       help="Specific episode indices for videos (default: first 5)")
    parser.add_argument("--output-dir", default="output/replay_videos",
                       help="Output directory for videos (default: output/replay_videos)")
    parser.add_argument("--fps", type=int, default=10,
                       help="Video framerate (default: 10)")
    parser.add_argument("--max-video-episodes", type=int, default=100,
                       help="Max episodes for video if --episodes not specified (default: 5)")
    parser.add_argument("--touch-only", action="store_true",
                       help="Generate only tactile sensor videos (no camera images)")
    
    args = parser.parse_args()
    
    # Analyze dataset structure first (always do this for basic info)
    zarr_file = analyze_zarr_structure(args.zarr_path)
    if zarr_file is None:
        return
    
    # Run inspection if requested
    if not args.video_only:
        inspect_dataset_detailed(
            zarr_path=zarr_file.store.path, # Use the resolved path
            show_samples=not args.no_samples,
            max_episodes=args.max_inspect_episodes
        )
    
    # Run video generation if requested
    if not args.inspect_only:
        replay_dataset_episodes(
            zarr_path=zarr_file.store.path, # Use the resolved path
            episode_indices=args.episodes,
            output_dir=args.output_dir,
            fps=args.fps,
            max_episodes=args.max_video_episodes,
            touch_only=args.touch_only
        )


if __name__ == "__main__":
    main()
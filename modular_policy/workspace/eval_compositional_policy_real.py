"""
Real-world evaluation script for compositional policy combining RGB and Tactile models.

Usage:
python modular_policy/workspace/eval_compositional_policy_real.py \
    --rgb_checkpoint <rgb_ckpt_path> \
    --tactile_checkpoint <tactile_ckpt_path> \
    --output <save_dir> \
    --robot_ip <ip>

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import torch
import dill
import pathlib
import skvideo.io
import hydra
from omegaconf import OmegaConf
import scipy.spatial.transform as st
import contextlib
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import defaultdict, deque
from modular_policy.real_world.real_ur5e_env import RealUR5eEnv
from modular_policy.real_world.spacemouse_shared_memory import Spacemouse
from modular_policy.common.precise_sleep import precise_wait
from modular_policy.common.trans_utils import interpolate_poses
from modular_policy.devices.tactile_sensor import DualTactileSensor

import diffusers
from modular_policy.real_world.real_util import (
    get_image_transform,
    policy_action_to_env_action,
    real_obs_to_policy_obs
)
from modular_policy.common.pytorch_util import dict_apply
from modular_policy.policy.base_policy import BasePolicy
from modular_policy.policy.dp_compositional import DiffusionCompositionalPolicy
from modular_policy.policy.dp_mode import MoDEPolicy
from modular_policy.workspace.eval_visualization import get_tactile_viz, create_camera_grid_viz

OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('--rgb_checkpoint', '-r', required=True, help='Path to RGB policy checkpoint', default = 'output/20250908/175602_train_dp_unet_rgb_marker-expert_N71/checkpoints/ep-1000_sr-0.999.ckpt')
@click.option('--tactile_checkpoint', '-t', required=True, help='Path to Tactile policy checkpoint', default = 'output/20250908/175553_train_dp_unet_tactile_marker-expert_N71/checkpoints/ep-2000_sr-1.000.ckpt')
@click.option('--rgb_weight', default=0.5, type=float, help='Weight for RGB policy in composition')
@click.option('--tactile_weight', default=0.5, type=float, help='Weight for Tactile policy in composition')
@click.option('--output', '-o', default='output/eval_compositional_marker', help='Directory to save recording')
@click.option('--robot_ip', default='192.168.0.2', help="UR5's IP address")
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--vis_cameras', default="0,1", type=str, help="Which RealSense cameras to visualize (e.g., '0,1,2' or 'all').")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=16, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=180, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.")
@click.option('--use_gripper', '-ug', default=True, type=bool)
@click.option('--human_ctrl_mode', '-hcm', default='eef', type=str)
@click.option('--policy_ctrl_mode', '-pcm', default='joint', type=str)
@click.option('--delta_action', '-da', default=False, type=bool)
@click.option('--action_offset', '-ao', default=0, type=int)
@click.option('--reset_robot', default=False, type=bool)
@click.option('--debug', '-d', default=False, type=bool)
@click.option('--use_tactile', '-ut', default=True, help="Use tactile sensors.")
@click.option('--show_tactile_viz', default=True, type=bool, help="Show tactile visualization window.")
@click.option('--tactile_left_port', default='/dev/ttyUSB1', help="Left tactile sensor port.")
@click.option('--tactile_right_port', default='/dev/ttyUSB0', help="Right tactile sensor port.")
@click.option('--tactile_scale_factor', default=25, type=int, help="Scale factor for tactile visualization size (default: 25)")
@click.option('--camera_window_width', default=1200, type=int, help="Camera window width in pixels (default: 1200)")
@click.option('--camera_window_height', default=900, type=int, help="Camera window height in pixels (default: 900)")

def main(rgb_checkpoint, tactile_checkpoint, rgb_weight, tactile_weight, output,
         robot_ip, match_dataset, match_episode,
         vis_cameras, init_joints, steps_per_inference, max_duration,
         frequency, command_latency, use_gripper, human_ctrl_mode,
         policy_ctrl_mode, delta_action, action_offset, reset_robot, debug,
         use_tactile, show_tactile_viz, tactile_left_port, tactile_right_port,
         tactile_scale_factor, camera_window_width, camera_window_height):
    
    # Validate weights
    assert abs(rgb_weight + tactile_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {rgb_weight + tactile_weight}"
    
    # Create output directory
    output_dir = pathlib.Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading RGB checkpoint from: {rgb_checkpoint}")
    print(f"Loading Tactile checkpoint from: {tactile_checkpoint}")
    print(f"Composition weights: RGB={rgb_weight}, Tactile={tactile_weight}")
    
    # Load RGB policy
    rgb_payload = torch.load(open(rgb_checkpoint, 'rb'), pickle_module=dill)
    rgb_cfg = rgb_payload['cfg']
    rgb_cls = hydra.utils.get_class(rgb_cfg._target_)
    rgb_workspace = rgb_cls(rgb_cfg, output_dir=output)
    rgb_policy = rgb_workspace.model
    rgb_policy.load_state_dict(rgb_payload['state_dicts']['model'])
    rgb_policy.eval()
    
    # Load Tactile policy  
    tactile_payload = torch.load(open(tactile_checkpoint, 'rb'), pickle_module=dill)
    tactile_cfg = tactile_payload['cfg']
    tactile_cls = hydra.utils.get_class(tactile_cfg._target_)
    tactile_workspace = tactile_cls(tactile_cfg, output_dir=output)
    tactile_policy = tactile_workspace.model
    tactile_policy.load_state_dict(tactile_payload['state_dicts']['model'])
    tactile_policy.eval()
    
    print("Successfully loaded both policies!")
    
    # BACKWARD COMPATIBILITY FIX: Add missing tactile observation ports
    # Only for policies that were specifically trained as tactile-only
    # Check policy name or config to determine if it's a tactile policy
    is_tactile_policy = False
    if hasattr(tactile_cfg, 'name') and 'tactile' in tactile_cfg.name:
        is_tactile_policy = True
    elif 'tactile' in str(tactile_checkpoint).lower():  # Check if checkpoint path contains 'tactile'
        is_tactile_policy = True
    
    if is_tactile_policy and hasattr(tactile_policy, 'modalities') and 'tactile' in tactile_policy.modalities:
        expected_tactile_keys = ['left_tactile', 'right_tactile']
        missing_tactile_keys = [key for key in expected_tactile_keys if key not in tactile_policy.obs_ports]
        if missing_tactile_keys:
            print(f"COMPATIBILITY FIX: Adding missing tactile observation ports: {missing_tactile_keys}")
            tactile_policy.obs_ports.extend(missing_tactile_keys)
            print(f"Updated tactile policy observation ports: {tactile_policy.obs_ports}")
    
    # Debug: Print policy information
    print(f"RGB policy observation ports: {rgb_policy.get_observation_ports()}")
    print(f"RGB policy normalizer keys: {list(rgb_policy.normalizer.params_dict.keys())}")
    print(f"Tactile policy observation ports: {tactile_policy.get_observation_ports()}")
    print(f"Tactile policy normalizer keys: {list(tactile_policy.normalizer.params_dict.keys())}")
    
    if debug:
        print(f"Debug mode enabled")
        print(f"Steps per inference: {steps_per_inference}")
        print(f"Max duration: {max_duration}s") 
        print(f"Control frequency: {frequency}Hz")
        print(f"Command latency: {command_latency}s")
        print(f"Use gripper: {use_gripper}")
        print(f"Human control mode: {human_ctrl_mode}")
        print(f"Policy control mode: {policy_ctrl_mode}")
        print(f"Delta action: {delta_action}")
        print(f"Reset robot: {reset_robot}")
    
    # Create compositional policy
    policy = DiffusionCompositionalPolicy(
        shape_meta=rgb_cfg.shape_meta,
        noise_scheduler=rgb_policy.noise_scheduler,  # Use same scheduler
        modular_policies=[rgb_policy, tactile_policy],
        policy_weights=[rgb_weight, tactile_weight],
        horizon=rgb_cfg.horizon,
        n_action_steps=rgb_cfg.n_action_steps,
        n_obs_steps=rgb_cfg.n_obs_steps,
        num_inference_steps=rgb_policy.num_inference_steps
    )
    
    # Patch the normalization function to handle observation/normalizer mismatch
    # RGB policy should only get: camera_0_color, camera_1_color, robot_joint
    # Tactile policy should only get: left_tactile, right_tactile, robot_joint
    
    def patched_normalization(obs_dict):
        nobs_list = []
        
        # RGB policy - only cameras and robot_joint
        rgb_valid_ports = [port for port in ['camera_0_color', 'camera_1_color', 'robot_joint'] 
                          if port in obs_dict]
        rgb_obs = {port: obs_dict[port] for port in rgb_valid_ports}
        rgb_nobs = rgb_policy.normalizer.normalize(rgb_obs)
        nobs_list.append(rgb_nobs)
        
        # Tactile policy - only tactile and robot_joint  
        tactile_valid_ports = [port for port in ['left_tactile', 'right_tactile', 'robot_joint'] 
                             if port in obs_dict]
        tactile_obs = {port: obs_dict[port] for port in tactile_valid_ports}
        tactile_nobs = tactile_policy.normalizer.normalize(tactile_obs)
        nobs_list.append(tactile_nobs)
        
        return nobs_list
    policy.normalization = patched_normalization
    
    # Also patch the observation encoder setup to handle the mismatch
    # The issue is that the compositional policy's obs_encoder expects certain observations
    # but we need to filter them properly
    
    print("Created compositional policy!")
    
    # Use the same task config as RGB (they should be identical)
    shape_meta = rgb_cfg.shape_meta
    
    # Parse vis_cameras argument
    if vis_cameras.lower() == 'all':
        # Assuming max 4 cameras for now, can be made more dynamic
        vis_camera_indices = list(range(4))
    else:
        try:
            vis_camera_indices = [int(x.strip()) for x in vis_cameras.split(',')]
        except ValueError:
            print(f"Error: Invalid camera indices specified in --vis_cameras: {vis_cameras}")
            return

    # load match_dataset
    match_camera_idx = 0
    episode_first_frame_map = dict()
    if match_dataset is not None:
        match_dir = pathlib.Path(match_dataset)
        match_video_dir = match_dir.joinpath('videos')
        for vid_dir in match_video_dir.glob("*/"):
            episode_idx = int(vid_dir.stem)
            match_video_path = vid_dir.joinpath(f'{match_camera_idx}.mp4')
            if match_video_path.exists():
                frames = skvideo.io.vread(
                    str(match_video_path), num_frames=1)
                episode_first_frame_map[episode_idx] = frames[0]
    print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")
        
    # Cameras setup
    n_cameras = len([k for k in shape_meta['obs'].keys() if k.startswith('camera_')])
    print(f"Using {n_cameras} cameras")
    
    # Setup environment parameters (copied from working eval_policy_real.py)
    num_joints = 6 + int(use_gripper)
    
    # Set inference params for the compositional policy
    policy.num_inference_steps = 16 # DDIM inference iterations
    policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    noise_scheduler = diffusers.schedulers.scheduling_ddim.DDIMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type='epsilon'
    )
    policy.noise_scheduler = noise_scheduler
    
    max_pos_speed=0.2
    max_rot_speed=0.4 

    # setup experiment
    dt = 1/frequency
    
    # Get observation resolution
    obs_res = None
    if obs_res is None:
        obs_res = (640, 480)  # current task does not have shape_meta

    real_obs_info = {}
    if 'camera_features' in shape_meta.obs:
        raise NotImplementedError("Camera features not supported yet.")

    n_obs_steps = rgb_cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("steps_per_inference:", steps_per_inference)
    print("action_offset:", action_offset)
    
    assert human_ctrl_mode == 'eef'
    assert use_gripper == True

    # Check if policy expects tactile data
    policy_expects_tactile = any('tactile' in port for port in policy.get_observation_ports())
    use_tactile_sensors = use_tactile and policy_expects_tactile
    
    if use_tactile and not policy_expects_tactile:
        print("Warning: --use_tactile specified but policy doesn't expect tactile data. Disabling tactile sensors.")
    elif not use_tactile and policy_expects_tactile:
        print("Warning: Policy expects tactile data but --use_tactile not specified. Will use dummy tactile data.")

    with SharedMemoryManager() as shm_manager:
        tactile_context = DualTactileSensor(tactile_left_port, tactile_right_port) if use_tactile_sensors else None
        
        with Spacemouse(shm_manager=shm_manager, use_gripper=use_gripper) as device, \
            (tactile_context if tactile_context else contextlib.nullcontext()) as tactile_sensors, \
            RealUR5eEnv(
                output_dir=output, 
                frequency=frequency,
                n_obs_steps=n_obs_steps,
                obs_image_resolution=obs_res,
                obs_float32=False,
                init_joints=init_joints,
                video_capture_fps=15, # 6, 15, 30
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                speed_slider_value=1,
                lookahead_time=0.1,
                gain=100, 
                enable_multi_cam_vis=False,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                single_arm_type='right',
                ctrl_mode=policy_ctrl_mode,
                use_gripper=use_gripper,
                tactile_sensors=tactile_sensors if use_tactile_sensors else None
            ) as env:
            
            cv2.setNumThreads(1)
            env.realsense.set_depth_preset('Default')
            env.realsense.set_depth_exposure(33000, 16)

            env.realsense.set_exposure(exposure=115, gain=64)
            env.realsense.set_contrast(contrast=60)
            env.realsense.set_white_balance(white_balance=3100)

            print("Waiting for realsense")
            time.sleep(3)

            print("Warming up policy inference")
            obs = env.get_obs()
            with torch.no_grad():
                policy.reset()
                # Determine vis_size from any available camera in shape_meta
                vis_size = (96, 128)  # default
                for key, attr in shape_meta['obs'].items():
                    if 'camera' in key and 'color' in key:
                        vis_size = attr['shape'][-2:]
                        break
                
                obs_dict_np = real_obs_to_policy_obs(
                    real_obs=obs,
                    vis_size=vis_size,
                )
                print(f"Available observation keys: {list(obs_dict_np.keys())}")
                print(f"Policy expects observation ports: {policy.get_observation_ports()}")
                
                # Filter observations based on what each individual policy can actually use
                # RGB policy: must be in both obs_ports AND normalizer  
                # Tactile policy: must be in normalizer (obs_ports may be incomplete due to training bug)
                
                rgb_policy_supported = [port for port in rgb_policy.get_observation_ports() 
                                      if port in rgb_policy.normalizer.params_dict and port in obs_dict_np]
                tactile_policy_supported = [port for port in tactile_policy.normalizer.params_dict 
                                          if port != 'action' and port in obs_dict_np]
                
                print(f"RGB policy supported: {rgb_policy_supported}")
                print(f"Tactile policy supported: {tactile_policy_supported}")
                
                # Combine and deduplicate
                all_supported = list(set(rgb_policy_supported + tactile_policy_supported))
                print(f"Using observation ports: {all_supported}")
                
                obs_dict_np = {
                    port: obs_dict_np[port]
                    for port in all_supported
                }
                def convert_to_tensor(x, key):
                    tensor = torch.from_numpy(x).unsqueeze(0).to(torch.device('cuda'))
                    # Convert camera images from uint8 to float only if they are uint8
                    if 'camera' in key and 'color' in key and x.dtype == np.uint8:
                        tensor = tensor.float() / 255.0
                    return tensor
                                        
                obs_dict = {key: convert_to_tensor(value, key) for key, value in obs_dict_np.items()}
                
                # Use torch.clamp instead of np.clip to keep tensors on GPU
                obs_dict['camera_0_color'] = torch.clamp(obs_dict['camera_0_color'] * 255, 0, 255).to(torch.uint8)
                obs_dict['camera_1_color'] = torch.clamp(obs_dict['camera_1_color'] * 255, 0, 255).to(torch.uint8)
                
                result = policy.predict_action(obs_dict)
                action = result['action'][0].detach().to('cpu').numpy()
                action = policy_action_to_env_action(raw_actions=action, 
                                                     cur_eef_pose_6d=obs['robot_eef_pose'][-1], 
                                                     action_mode=policy_ctrl_mode)
                del result
                
            intermediate_pose = []
            intermediate_joints = []
            gripper_pose = 0

            # reset robot
            if reset_robot:
                print('Resetting robot...')
                env.robot.switch_mode('joint')
                env.robot.set_robot_joints(np.deg2rad([0, -90, -90, -90, 90, 0]), speed=0.01)
            
            print('Ready!')
            
            # Initialize OpenCV windows with resizable capability
            cv2.namedWindow('default', cv2.WINDOW_NORMAL)
            
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                state = env.get_robot_state()
                target_pose = state['TargetTCPPose']
                t_start = time.monotonic()
                iter_idx = 0
                while True:

                    # Calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt 

                    # pump obs
                    obs = env.get_obs()

                    # Visualize
                    episode_id = env.replay_buffer.n_episodes
                    
                    # Create camera grid visualization
                    vis_img = create_camera_grid_viz(obs, vis_camera_indices)

                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = np.minimum(vis_img, match_img)

                    # Create tactile visualization if enabled
                    tactile_viz = None
                    if show_tactile_viz and use_tactile_sensors and 'left_tactile' in obs and 'right_tactile' in obs:
                        tactile_viz = get_tactile_viz(obs, tactile_scale_factor)

                    # Combine camera and tactile visualizations in a two-column layout
                    if vis_img is not None and tactile_viz is not None:
                        # --- Main setting: Define the height for the final combined image ---
                        # Both images will be scaled to this height while preserving their aspect ratio.
                        # You can change this value to make the overall visualization larger or smaller.
                        common_height = 800  # pixels

                        # Get original dimensions and calculate aspect ratios
                        rgb_h, rgb_w, _ = vis_img.shape
                        tactile_h, tactile_w, _ = tactile_viz.shape
                        
                        rgb_aspect_ratio = rgb_w / rgb_h
                        tactile_aspect_ratio = tactile_w / tactile_h
                        
                        # Calculate the new widths that preserve the aspect ratio for the common height
                        rgb_new_width = int(common_height * rgb_aspect_ratio)
                        tactile_new_width = int(common_height * tactile_aspect_ratio)
                        
                        # Resize both images to the new dimensions
                        rgb_resized = cv2.resize(vis_img, (rgb_new_width, common_height), interpolation=cv2.INTER_AREA)
                        tactile_resized = cv2.resize(tactile_viz, (tactile_new_width, common_height), interpolation=cv2.INTER_AREA)
                        
                        # Combine the two images horizontally. This is clean and avoids black bars
                        # because both images now have the exact same height.
                        # We place the tactile visualization on the left and the camera on the right.
                        vis_img = np.hstack([tactile_resized, rgb_resized])

                    elif tactile_viz is not None:
                        # If there is no camera image, just show the tactile visualization
                        vis_img = tactile_viz

                    # Resize final output window if needed, BUT without distorting the image.
                    # This block replaces your final `cv2.resize`. It will shrink the image
                    # to fit into the target dimensions if it's too large, while maintaining the correct aspect ratio.
                    if vis_img is not None:
                        # Set your desired maximum window size here
                        max_display_width = 1600
                        max_display_height = 900

                        # Get the current dimensions of the combined image
                        img_h, img_w, _ = vis_img.shape

                        # Calculate the ratio to scale down the image
                        scale = min(max_display_width / img_w, max_display_height / img_h)

                        # Only resize if the image is larger than the max dimensions
                        if scale < 1.0:
                            new_w = int(img_w * scale)
                            new_h = int(img_h * scale)
                            vis_img = cv2.resize(vis_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    
                    
                    # Keyboard control
                    key_stroke = cv2.pollKey()
                    # Exit program
                    if key_stroke == ord('q'):
                        env.end_episode()
                        exit(0)
                    # Exit human control loop; hand control over to the policy
                    elif key_stroke == ord('c'):
                        break
                    # go home 
                    elif key_stroke == ord('h'): 
                        state = env.get_robot_state()
                        start_pose = state['ActualTCPPose']
                        final_pose = np.array([0.5268, -0.013,  0.275,  0, 3.141, 0, state['ActualTCPPose'][6]])
                        
                        # Define "Up" pose: lift to final Z height while maintaining current X,Y position
                        up_pose = start_pose.copy()
                        up_pose[2] = final_pose[2]  # Set Z-coordinate to final pose Z-coordinate
                        
                        # First movement (Up): interpolate from start_pose to up_pose
                        first_trajectory, _, _ = interpolate_poses(start_pose, up_pose, 0.005)
                        
                        # Second movement (To Goal): interpolate from up_pose to final_pose
                        second_trajectory, _, _ = interpolate_poses(up_pose, final_pose, 0.005)
                        
                        # Combine trajectories: concatenate waypoints from both movements
                        intermediate_pose.extend(first_trajectory)
                        intermediate_pose.extend(second_trajectory)
                        intermediate_pose.append(final_pose)

                    precise_wait(t_sample)
                    
                    # SpaceMouse control
                    if device.is_button_pressed(0) and not device.is_button_pressed(1):
                        # Left button pressed: close gripper
                        gripper_pose = 1
                    elif not device.is_button_pressed(0) and device.is_button_pressed(1):
                        # Right button pressed: open gripper
                        gripper_pose = 0
                    device_state = device.get_motion_state_transformed()
                    # print(f"Device state: {device_state}, gripper: {gripper_pose}")
                    
                    dpos = device_state[:3] * (env.max_pos_speed / frequency)
                    # Set rotation to zero
                    drot_xyz = device_state[3:] * 0
                    # drot_xyz = device_state[3:] * (env.max_rot_speed / frequency)
                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    
                    if intermediate_pose != []:
                        target_pose = intermediate_pose.pop(0)
                    else:
                        target_pose[0:3] += dpos
                        target_pose[3:6] = (drot * st.Rotation.from_rotvec(
                            target_pose[3:6])).as_rotvec()
                        target_pose[6] = gripper_pose

                    target_joints = np.zeros((num_joints,))

                    # execute teleop command
                    env.exec_actions(
                        joint_actions=[target_joints], 
                        eef_actions=[target_pose],
                        mode=human_ctrl_mode,     # HACK: always use eef mode for space mouse
                        timestamps=[t_command_target-time.monotonic()+time.time()])
                    


                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    term_area_start_timestamp = float('inf')
                    prev_target_pose = None
                    prev_target_joints = None
                    
                    while True:

                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        # print(f'Obs latency {time.time() - obs_timestamps[-1]}')
                  
                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            # Determine vis_size from any available camera in shape_meta
                            vis_size = (96, 128)  # default
                            for key, attr in shape_meta['obs'].items():
                                if 'camera' in key and 'color' in key:
                                    vis_size = attr['shape'][-2:]
                                    break
                            
                            obs_dict_np = real_obs_to_policy_obs(
                                real_obs=obs,
                                vis_size=vis_size,
                            )
                            
                            # Filter observations based on what each individual policy can actually use
                            # RGB policy: must be in both obs_ports AND normalizer
                            # Tactile policy: must be in normalizer (obs_ports may be incomplete due to training bug)
                            
                            rgb_policy_supported = [port for port in rgb_policy.get_observation_ports() 
                                                  if port in rgb_policy.normalizer.params_dict and port in obs_dict_np]
                            tactile_policy_supported = [port for port in tactile_policy.normalizer.params_dict 
                                                      if port != 'action' and port in obs_dict_np]
                            
                            print(f"Available observations: {list(obs_dict_np.keys())}")
                            print(f"RGB policy supported: {rgb_policy_supported}")
                            print(f"Tactile policy supported: {tactile_policy_supported}")
                            
                            # Combine and deduplicate
                            all_supported = list(set(rgb_policy_supported + tactile_policy_supported))
                            print(f"Using observations: {all_supported}")
                            
                            obs_dict_np = {
                                port: obs_dict_np[port]
                                for port in all_supported
                            }
                            
                            obs_dict = {key: torch.from_numpy(value).unsqueeze(0).to(torch.device('cuda'))  for key, value in obs_dict_np.items()}

                            result = policy.predict_action(obs_dict)
                            # this action starts from the first obs step
                            action = result['action'][0].detach().to('cpu').numpy()
                            action = policy_action_to_env_action(raw_actions=action, 
                                                                cur_eef_pose_6d=obs['robot_eef_pose'][-1], 
                                                                action_mode=policy_ctrl_mode)

                            # print('Inference latency:', time.time() - s)
                        
                        # convert policy action to env actions
                        if policy_ctrl_mode == 'eef':
                            if delta_action:
                                assert len(action) == 1
                                if prev_target_pose is None:
                                    prev_target_pose = obs['robot_eef_pose'][-1]
                                this_target_pose = prev_target_pose.copy()
                                this_target_pose[[0,1]] += action[-1]
                                prev_target_pose = this_target_pose
                                this_target_poses = np.expand_dims(this_target_pose, axis=0)
                            else:
                                this_target_poses = action
                        elif policy_ctrl_mode == 'joint':
                            if delta_action:
                                if prev_target_joints is None:
                                    prev_target_joints = obs['robot_joint'][-1]
                                this_target_joints = np.cumsum(action, axis=0) + prev_target_joints.copy()
                                prev_target_joints = this_target_joints
                            else:
                                this_target_joints = action
                        
                        
                        
                        
                        
                        # deal with timing
                        # the same step actions are always the target for
                        action_timestamps = (np.arange(len(action), dtype=np.float64) + action_offset
                            ) * dt + obs_timestamps[-1]
                        action_exec_latency = 0.01
                        curr_time = time.time()
                        is_new = action_timestamps > (curr_time + action_exec_latency)

                        if policy_ctrl_mode == 'eef':
                            if np.sum(is_new) == 0:
                                # exceeded time budget, still do something
                                this_target_poses = this_target_poses[[-1]]
                                # schedule on next available step
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]
                        elif policy_ctrl_mode == 'joint':
                            pass

                        # execute actions
                        if policy_ctrl_mode == 'eef':
                            this_target_joints = [np.zeros(num_joints) for _ in this_target_poses]
                            this_target_joints = np.stack(this_target_joints, axis=0)
                        elif policy_ctrl_mode == 'joint':
                            this_target_poses = [np.zeros(6 + int(use_gripper)) for _ in this_target_joints]
                            this_target_poses = np.stack(this_target_poses, axis=0)
                        
                        env.exec_actions(
                            joint_actions=this_target_joints, 
                            eef_actions=this_target_poses,
                            mode=policy_ctrl_mode,
                            timestamps=action_timestamps)
                        
                        
                        # for eef_joints in this_target_joints[0:8]:
                        #     env.robot.servoJ(eef_joints, duration=0.1)
                        #     time.sleep(0.01)

                        print(f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        
                        # Create camera grid visualization
                        vis_img = create_camera_grid_viz(obs, vis_camera_indices)

                        match_episode_id = episode_id
                        if match_episode is not None:
                            match_episode_id = match_episode
                        if match_episode_id in episode_first_frame_map:
                            match_img = episode_first_frame_map[match_episode_id]
                            ih, iw, _ = match_img.shape
                            oh, ow, _ = vis_img.shape
                            tf = get_image_transform(
                                input_res=(iw, ih), 
                                output_res=(ow, oh), 
                                bgr_to_rgb=False)
                            match_img = tf(match_img).astype(np.float32) / 255
                            vis_img = np.minimum(vis_img, match_img)

                        # Create tactile visualization if enabled
                        tactile_viz = None
                        if show_tactile_viz and use_tactile_sensors and 'left_tactile' in obs and 'right_tactile' in obs:
                            tactile_viz = get_tactile_viz(obs, tactile_scale_factor)

                        # Combine camera and tactile visualizations in a two-column layout
                        if vis_img is not None and tactile_viz is not None:
                            # --- Main setting: Define the height for the final combined image ---
                            # Both images will be scaled to this height while preserving their aspect ratio.
                            # You can change this value to make the overall visualization larger or smaller.
                            common_height = 800  # pixels

                            # Get original dimensions and calculate aspect ratios
                            rgb_h, rgb_w, _ = vis_img.shape
                            tactile_h, tactile_w, _ = tactile_viz.shape
                            
                            rgb_aspect_ratio = rgb_w / rgb_h
                            tactile_aspect_ratio = tactile_w / tactile_h
                            
                            # Calculate the new widths that preserve the aspect ratio for the common height
                            rgb_new_width = int(common_height * rgb_aspect_ratio)
                            tactile_new_width = int(common_height * tactile_aspect_ratio)
                            
                            # Resize both images to the new dimensions
                            rgb_resized = cv2.resize(vis_img, (rgb_new_width, common_height), interpolation=cv2.INTER_AREA)
                            tactile_resized = cv2.resize(tactile_viz, (tactile_new_width, common_height), interpolation=cv2.INTER_AREA)
                            
                            # Combine the two images horizontally. This is clean and avoids black bars
                            # because both images now have the exact same height.
                            # We place the tactile visualization on the left and the camera on the right.
                            vis_img = np.hstack([tactile_resized, rgb_resized])

                        elif tactile_viz is not None:
                            # If there is no camera image, just show the tactile visualization
                            vis_img = tactile_viz

                        # Resize final output window if needed, BUT without distorting the image.
                        # This block replaces your final `cv2.resize`. It will shrink the image
                        # to fit into the target dimensions if it's too large, while maintaining the correct aspect ratio.
                        if vis_img is not None:
                            # Set your desired maximum window size here
                            max_display_width = 1600
                            max_display_height = 900

                            # Get the current dimensions of the combined image
                            img_h, img_w, _ = vis_img.shape

                            # Calculate the ratio to scale down the image
                            scale = min(max_display_width / img_w, max_display_height / img_h)

                            # Only resize if the image is larger than the max dimensions
                            if scale < 1.0:
                                new_w = int(img_w * scale)
                                new_h = int(img_h * scale)
                                vis_img = cv2.resize(vis_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        
                        time_spent = time.monotonic() - t_start
                        
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time_spent
                        )
                        
                        cv2.imshow('default', vis_img[...,::-1])
                        

                        key_stroke = cv2.pollKey()
                        if key_stroke == ord('s'):
                            # Stop episode
                            # Hand control back to human
                            print("time_spent", time_spent)
                            env.end_episode()
                            print('Stopped.')
                            break

                        # auto termination
                        terminate = False
                        if time.monotonic() - t_start > max_duration:
                            terminate = True
                            print('Terminated by the timeout!')

                        if terminate:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference
                        
                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()

                print("Stopped.")
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
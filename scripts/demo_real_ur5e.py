"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start recording.
Press "S" to stop recording.
Press "Q" to exit program.
Press "Backspace" to delete the previously recorded episode.
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import contextlib
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import yaml
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modular_policy.real_world.real_ur5e_env import RealUR5eEnv
from modular_policy.real_world.spacemouse_shared_memory import Spacemouse
from modular_policy.devices.tactile_sensor import DualTactileSensor
from modular_policy.common.precise_sleep import precise_sleep, precise_wait
from modular_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from modular_policy.common.trans_utils import interpolate_poses

def get_vis(skip_cam_vis, obs, vis_camera_idx, in_hand_camera_idx=None, use_tactile=False):
    
    if not skip_cam_vis:
        # visualize
        vis_color_img = obs[f'camera_{vis_camera_idx}_color'][-1, :, :, ::-1].copy()

        if in_hand_camera_idx is not None:
            in_hand_color_img = obs[f'camera_{in_hand_camera_idx}_color'][-1, :, :, ::-1].copy()

            # Stack in-hand camera images horizontally
            vis_color_img = np.hstack((vis_color_img, in_hand_color_img))
    else:
        vis_color_img = None
    # Create tactile visualization if enabled (following 3D-ViTac approach)
    tactile_viz = None
    if use_tactile and 'left_tactile' in obs and 'right_tactile' in obs:
        # Get most recent tactile data
        left_tactile = obs['left_tactile'][-1][0]
        right_tactile = obs['right_tactile'][-1][0]
        
        # Scale to 0-255 (3D-ViTac approach)
        left_scaled = (left_tactile * 255).astype(np.uint8)
        right_scaled = (right_tactile * 255).astype(np.uint8)
        
        # Apply VIRIDIS colormap (3D-ViTac style)
        left_colored = cv2.applyColorMap(left_scaled, cv2.COLORMAP_VIRIDIS)
        right_colored = cv2.applyColorMap(right_scaled, cv2.COLORMAP_VIRIDIS)
        
        # Use 3D-ViTac scaling approach (30x scale factor, adjusted for display)
        scale_factor = 15  # Adjusted for display alongside camera
        left_resized = cv2.resize(left_colored, 
                                (left_tactile.shape[1] * scale_factor, left_tactile.shape[0] * scale_factor),
                                interpolation=cv2.INTER_NEAREST)
        right_resized = cv2.resize(right_colored, 
                                (right_tactile.shape[1] * scale_factor, right_tactile.shape[0] * scale_factor),
                                interpolation=cv2.INTER_NEAREST)
        
        # Add labels (3D-ViTac style)
        cv2.putText(left_resized, 'Left Tactile', (5, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(right_resized, 'Right Tactile', (5, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Stack tactile visualizations vertically
        tactile_viz = np.vstack([left_resized, right_resized])
    
    # Combine all visualizations
    if vis_color_img is None:
        return tactile_viz
    if tactile_viz is not None:
        # Pad tactile visualization to match camera height if needed
        if tactile_viz.shape[0] < vis_color_img.shape[0]:
            pad_height = vis_color_img.shape[0] - tactile_viz.shape[0]
            padding = np.zeros((pad_height, tactile_viz.shape[1], 3), dtype=np.uint8)
            tactile_viz = np.vstack([tactile_viz, padding])
        elif tactile_viz.shape[0] > vis_color_img.shape[0]:
            tactile_viz = tactile_viz[:vis_color_img.shape[0], :, :]
        
        vis_img = np.concatenate((vis_color_img, tactile_viz), axis=1)
    else:
        vis_img = np.concatenate((vis_color_img), axis=1)
    
    return vis_img

WAIT_Z = 0.2
@click.command()
@click.option('--output', '-o', default='demo_data_raw/marker_expert_20_raw_new', help="Directory to save demonstration dataset.")
@click.option('--robot_left_ip', '-ri', default='192.168.0.3', help="UR5's IP address e.g. 192.168.0.204")
@click.option('--robot_right_ip', '-ri', default='192.168.0.2', help="UR5's IP address e.g. 192.168.0.204")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--in_hand_camera_idx', default=2, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=True, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('--debug', '-d', is_flag=True, default=False, help="Debug mode.")
@click.option('--dummy_robot', '-dr', is_flag=True, default=False, help="Dummy robot.")
@click.option('--count_time', '-ct', is_flag=True, default=False, help="Count the time to execute the program.", type=bool)
@click.option('--input_device', '-id', default='spacemouse', type=click.Choice(['spacemouse', 'gello'], case_sensitive=False), help="spacemouse")
@click.option('--ctrl_mode', '-cm', default='eef')
@click.option('--use_gripper', '-ug', default=True, help="Use gripper or not.")
@click.option('--robot_type', '-rt', default='ur5e')
@click.option('--use_tactile', '-ut', default=True, help="Use tactile sensors.")
@click.option('--tactile_left_port', default='/dev/ttyUSB1', help="Left tactile sensor port.")
@click.option('--tactile_right_port', default='/dev/ttyUSB0', help="Right tactile sensor port.")

def main(output, robot_left_ip, robot_right_ip, vis_camera_idx, in_hand_camera_idx, init_joints, frequency, command_latency,
         debug, dummy_robot, count_time, input_device, ctrl_mode, use_gripper, robot_type,
         use_tactile, tactile_left_port, tactile_right_port):
    
    # assert input_device == 'spacemouse'
    assert robot_type == 'ur5e'
    assert ctrl_mode == 'eef'
    
    max_pos_speed=0.2
    max_rot_speed=0.4 

    # dynamixel control box port map (to distinguish left and right gello)
    gello_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modular_policy/devices/gello_software/gello.yaml")
    with open(gello_config_path) as stream:
        try:
            gello_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    single_robot_type = gello_config['single_arm_type']
    reset_joints = np.deg2rad(gello_config[f"{single_robot_type}_reset_joints"])
    dt = 1/frequency

    # Attempt to get the device driver class from the dictionary using the input_device key
    device_driver = Spacemouse

    robot_env = RealUR5eEnv

    with SharedMemoryManager() as shm_manager:
        tactile_context = DualTactileSensor(tactile_left_port, tactile_right_port) if use_tactile else None
        
        with KeystrokeCounter() as key_counter, \
            device_driver(shm_manager=shm_manager,
                          use_gripper=use_gripper) as device, \
            (tactile_context if tactile_context else contextlib.nullcontext()) as tactile_sensors, \
            robot_env(
                output_dir=output, 
                # recording resolution
                obs_image_resolution=(640, 480), # (1280, 720), (640, 480), (480, 270)
                frequency=frequency,
                init_joints=init_joints,
                j_init=reset_joints,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                speed_slider_value=1,
                enable_multi_cam_vis=False,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager,
                enable_depth=True, # TODO: when set to True, it makes the robot show jittery behavior and color image fronzen
                debug=debug,
                dummy_robot=dummy_robot,
                single_arm_type=single_robot_type,
                ctrl_mode=ctrl_mode,
                use_gripper=use_gripper,
                tactile_sensors=tactile_sensors
            ) as env:
            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_depth_preset('Default')
            env.realsense.set_depth_exposure(33000, 16)

            env.realsense.set_exposure(exposure=115, gain=64)
            env.realsense.set_contrast(contrast=60)
            env.realsense.set_white_balance(white_balance=3100)

            obs_duration = 0.0

            time.sleep(5.0)
            print("Number of cameras: ", env.realsense.n_cameras)
            print('Ready!')
            
            state = env.get_robot_state()
            if not dummy_robot:
                target_pose = state['TargetTCPPose']

            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            intermediate_pose = []
            gripper_pose = 0

            cur_dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
            robot_extrinsics_path = os.path.join(cur_dir_path, 'modular_policy/real_world/robot_extrinsics')
            robots = ['left', 'right']

            robot_base_in_world = {}
            for robot in robots:
                robot_base_in_world[robot] = np.load(os.path.join(robot_extrinsics_path, f'{robot}_base_pose_in_world.npy'))
            
            while not stop:

                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * (dt+obs_duration) # the end time of the current control cycle
                t_sample = t_cycle_end - command_latency # the time when the system should sample spacemouse input
                t_command_target = t_cycle_end + dt # the future time when the the target pose should be reached
                
                # get obs
                obs = env.get_obs()
                
                # get gripper input
                if device.is_button_pressed(0) and not device.is_button_pressed(1):
                    gripper_pose = 1 # Left button pressed: close gripper
                elif not device.is_button_pressed(0) and device.is_button_pressed(1):
                    gripper_pose = 0 # Right button pressed: open gripper

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                        # go home 
                        state = env.get_robot_state()
                        start_pose = state['ActualTCPPose']
                        final_pose = np.array([0.5268, -0.013,  0.275,  0, 3.141, 0, 0])
                        step_intermediate_pose, _, _ = interpolate_poses(start_pose, final_pose, 0.003)
                        intermediate_pose.extend(step_intermediate_pose)
                        intermediate_pose.append(final_pose)
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                    elif key_stroke == KeyCode(char='t'):
                        # go to top position
                        state = env.get_robot_state()
                        start_pose = state['ActualTCPPose']
                        final_pose = np.array([-0.13451088, -0.64240012,  0.48906811,  1.5714615 ,  0.00809301, -0.00683493, 0])
                        step_intermediate_pose, _, _ = interpolate_poses(start_pose, final_pose, 0.005)
                        intermediate_pose.extend(step_intermediate_pose)
                        intermediate_pose.append(final_pose)
                        key_counter.clear()
                    elif key_stroke == KeyCode(char='h'):
                        # go home 
                        state = env.get_robot_state()
                        start_pose = state['ActualTCPPose']
                        final_pose = np.array([0.5268, -0.013,  0.275,  0, 3.141, 0, 0])
                        step_intermediate_pose, _, _ = interpolate_poses(start_pose, final_pose, 0.005)
                        intermediate_pose.extend(step_intermediate_pose)
                        intermediate_pose.append(final_pose)
                        key_counter.clear()
                stage = key_counter[Key.space]

                # visualize
                episode_id = env.replay_buffer.n_episodes
                vis_img = get_vis(skip_cam_vis=False, obs=obs, vis_camera_idx=vis_camera_idx, in_hand_camera_idx=in_hand_camera_idx, use_tactile=use_tactile)
                text = f'Episode: {episode_id}, Stage: {stage}'
                if is_recording:
                    text += ', Recording!'
                cv2.putText(
                    vis_img,
                    text,
                    (10,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=2,
                    color=(255,255,255)
                )
                cv2.imshow('default', vis_img)
                cv2.pollKey()
                
                precise_wait(t_sample)
                
                # get teleop command
                device_state = device.get_motion_state_transformed()
                dpos = device_state[:3] * (env.max_pos_speed / frequency)
                # # Set rotation to zero
                # drot_xyz = device_state[3:] * 0
                # Set z rotation to zero
                drot_xyz = device_state[3:]
                drot_xyz[0:2] = drot_xyz[0:2] * 0
                drot_xyz = drot_xyz * (env.max_rot_speed / frequency)
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                
                # execute teleop command
                if not dummy_robot:
                    if intermediate_pose != []:
                        target_pose = intermediate_pose.pop(0)
                    else:
                        target_pose[:3] += dpos
                        target_pose[3:6] = (drot * st.Rotation.from_rotvec(
                            target_pose[3:6])).as_rotvec()
                    if use_gripper:
                        target_pose[6] = gripper_pose
                    env.exec_actions(
                        joint_actions=[np.zeros((6,))], 
                        eef_actions=[target_pose],
                        mode='eef',
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        stages=[stage])

                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()

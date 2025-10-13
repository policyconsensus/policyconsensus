from typing import Optional
import pathlib
import numpy as np
import time
import shutil
import math
import cv2
import omegaconf
import open3d as o3d
import os
import matplotlib.pyplot as plt
import cProfile
import pstats
import io
import json
import yaml
import glob
import re
import cv2  
import warnings
from filelock import FileLock
from scipy.spatial.transform import Rotation as R


from multiprocessing.managers import SharedMemoryManager
import sys
sys.path.insert(1, '.')
from os.path import dirname, abspath
import zarr
from modular_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from modular_policy.real_world.video_recorder import VideoRecorder
from modular_policy.common.timestamp_accumulator import TimestampObsAccumulator,TimestampActionAccumulator,align_timestamps
from modular_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from modular_policy.common.replay_buffer import ReplayBuffer
from modular_policy.common.cv2_util import get_image_transform, optimal_row_cols
from modular_policy.model.common.tensor_util import index_at_time
from modular_policy.real_world.rtde_interpolation_controller import RTDEInterpolationController
from modular_policy.common.data_utils import save_dict_to_hdf5, load_dict_from_hdf5
from modular_policy.common.precise_sleep import precise_wait
from modular_policy.model.common.rotation_transformer import RotationTransformer
from modular_policy.common.cv2_util import get_image_transform
from modular_policy.common.trans_utils import are_joints_close
from modular_policy.common.data_utils import  policy_action_to_env_action, _homo_to_9d_action, _9d_to_homo_action, _6d_axis_angle_to_homo, _homo_to_6d_axis_angle



DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',
    # 'gripper_pos': 'robot_gripper_pos',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp',
    'FtRawWrench': 'robot_ft_wrench',
    'robot_base_pose_in_world': 'robot_base_pose_in_world',
}


class RealUR5eEnv:
    def __init__(self,
                 # required params
                 output_dir,
                 robot_left_ip='192.168.0.3',
                 robot_right_ip='192.168.0.2',
                 # env params
                 frequency=10,
                 n_obs_steps=2,
                 # obs
                 obs_image_resolution=(640, 480),
                 max_obs_buffer_size=30,
                 camera_serial_numbers=None,
                 obs_key_map=DEFAULT_OBS_KEY_MAP,
                 obs_float32=False,
                 # action
                 max_pos_speed=0.05,
                 max_rot_speed=0.1,
                 speed_slider_value=0.1,
                 lookahead_time=0.2,
                 gain=100, # TODO: check this, 100
                 # robot
                 tcp_offset=0.13,
                 init_joints=False,
                 j_init=None,
                 # video capture params
                 video_capture_fps=15, # 6, 15, 30
                 video_capture_resolution=(640, 480),
                 # saving params
                 record_raw_video=True,
                 thread_per_video=2,
                 video_crf=21,
                 # vis params
                 enable_multi_cam_vis=False,
                 multi_cam_vis_resolution=(640, 480),
                 # shared memory
                 shm_manager=None,
                 enable_depth=True,
                 debug=False,
                 dummy_robot=False,
                 enable_pose=True,
                 single_arm_type='right',
                 ctrl_mode='joint',
                 use_gripper=True,
                 # tactile sensor params
                 tactile_sensors=None,
                 ):
        
        video_capture_resolution = obs_image_resolution
        multi_cam_vis_resolution = obs_image_resolution
        
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        if not output_dir.parent.is_dir():
            print(f"Output directory {output_dir.parent} does not exist! Creating...")
            output_dir.parent.mkdir(parents=True, exist_ok=True)

        # assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)

        self.save_format = 'zarr' # hdf5 or zarr

        if self.save_format == 'zarr':
            self.episode_id = 0
        else:
            self.episode_id = len(glob.glob(os.path.join(output_dir.absolute().as_posix(), '*.hdf5')))

        
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')
        self.debug = debug
        self.dummy_robot = dummy_robot
        self.enable_depth = enable_depth
        self.enable_pose = enable_pose
        self.tactile_sensors = tactile_sensors

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        color_tf = get_image_transform(
            input_res=video_capture_resolution,
            output_res=obs_image_resolution,
            # obs output rgb
            bgr_to_rgb=True)
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data['color'] = color_transform(data['color'])
            # data['depth'] = refine_depth_image(data['depth'])
            return data

        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(camera_serial_numbers),
            in_wh_ratio=obs_image_resolution[0]/obs_image_resolution[1],
            max_resolution=multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw,rh),
            bgr_to_rgb=False
        )

        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transfrom = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transfrom = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'
            
        # video_zarr_path = [str(output_dir.joinpath(f'camera_{i}.zarr').absolute()) for i in range(len(camera_serial_numbers))]
        video_zarr_path = str(output_dir.joinpath(f'videos.zarr').absolute())
        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec='h264',
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video,
            video_capture_resolution=video_capture_resolution,
            video_zarr_path=video_zarr_path,
            num_cams=len(camera_serial_numbers),
        )
        # self.video_recorder = video_recorder

        

        if self.debug:
            print("Initializing RealSense cameras...")
        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            # send every frame immediately after arrival
            # ignores put_fps
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=enable_depth,
            enable_infrared=False,
            enable_pointcloud=False,
            process_pcd=False,
            draw_keypoints=False,
            keypoint_kwargs=None,
            get_max_k=max_obs_buffer_size,
            # TODO: check why is transform and vis_transform blocking the program
            transform=transform,
            # vis_transform=vis_transform if enable_multi_cam_vis else None, # TODO: it is blocking the program
            # recording_transform=recording_transfrom,
            video_recorder=video_recorder,
            verbose=False,
        )
        if self.debug:
            print("RealSense cameras initialized!")


        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )

        cube_diag = np.linalg.norm([1, 1, 1])
        # j_init = np.array([-140, -120, -95, -150, -55, -180,
        #                    30, -120, -95, -150, -55, 180]) / 180 * np.pi
        if not init_joints:
            j_init = None

        robot_ip = robot_left_ip if single_arm_type == 'left' else robot_right_ip
        robot = RTDEInterpolationController(
            shm_manager=shm_manager,
            robot_ip=robot_ip,
            frequency=125,  # UR5 CB3 RTDE
            lookahead_time=0.2,
            gain=100, # TODO: check this, 100
            max_pos_speed=max_pos_speed*cube_diag,
            max_rot_speed=max_rot_speed*cube_diag,
            speed_slider_value=speed_slider_value,
            launch_timeout=3,
            tcp_offset_pose=None, #[0, 0, tcp_offset, 0, 0, 0],
            payload_mass=None,
            payload_cog=None,
            joints_init=j_init,
            joints_init_speed=0.5, # 1.05,
            soft_real_time=False,
            # verbose=False,
            receive_keys=None,
            get_max_k=max_obs_buffer_size,
            dummy_robot=self.dummy_robot,
            ctrl_mode=ctrl_mode,
            use_gripper=use_gripper,
        )
        self.realsense = realsense
        self.robot = robot
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.obs_key_map = obs_key_map
        self.use_gripper = use_gripper
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        if self.save_format == 'zarr':
            self.episode_id = replay_buffer.n_episodes
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        # self.action_accumulator = None
        self.joint_action_accumulator = None
        self.eef_action_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
        
        
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        if self.debug:
            print("Starting RealPushLEnv...")
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()
        if self.debug:
            print("RealPushLEnv started!")

    def stop(self, wait=True):
        self.end_episode()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self, skip_keypoint=False, profile=False) -> dict:
        "observation dict"
        assert self.is_ready
        if profile:
            # Initialize profiler
            pr = cProfile.Profile()
            pr.enable()
        
        # get data
        # 30 Hz, camera_receive_timestamp
        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency)) # how many of the most recent observations from the camera we want to fetch
        self.last_realsense_data = self.realsense.get(
            k=k, 
            out=self.last_realsense_data)
        
        # 125 hz, robot_receive_timestamp
        last_robot_data = self.robot.get_all_state()
        # both have more than n_obs_steps data

        # align camera obs timestamps
        dt = 1 / self.frequency
        last_timestamp = np.max([x['timestamp'][-1] for x in self.last_realsense_data.values()]) # the most recent timestamp from the collected camera data
        obs_align_timestamps = last_timestamp - (np.arange(self.n_obs_steps)[::-1] * dt)


        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            # remap key
            camera_obs[f'camera_{camera_idx}_color'] = value['color'][this_idxs]
            camera_obs[f'camera_{camera_idx}_depth'] = value['depth'][this_idxs]
            camera_obs[f'camera_{camera_idx}_intrinsics'] = value['intrinsics'][this_idxs]
            camera_obs[f'camera_{camera_idx}_extrinsics'] = value['extrinsics'][this_idxs]



        robot_timestamps = last_robot_data['robot_receive_timestamp']
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = dict()
        for k, v in last_robot_data.items():
            if k in self.obs_key_map:
                robot_obs_raw[self.obs_key_map[k]] = v
        
        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            robot_obs[k] = v[this_idxs]

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        
        # Add tactile data if available
        if self.tactile_sensors is not None and self.tactile_sensors.is_ready():
            tactile_data = self.tactile_sensors.get_contact_data()
            print(f"DEBUG: Got tactile data - Left shape: {tactile_data['left_tactile'].shape}, Right shape: {tactile_data['right_tactile'].shape}")
            # Repeat tactile data for n_obs_steps to match other observations
            obs_data['left_tactile'] = np.repeat(np.expand_dims(tactile_data['left_tactile'][np.newaxis, :, :], axis=0), self.n_obs_steps, axis=0)
            obs_data['right_tactile'] = np.repeat(np.expand_dims(tactile_data['right_tactile'][np.newaxis, :, :], axis=0), self.n_obs_steps, axis=0)
            # obs_data['left_tactile'] = np.repeat(tactile_data['left_tactile'][np.newaxis, :, :], self.n_obs_steps, axis=0)
            # obs_data['right_tactile'] = np.repeat(tactile_data['right_tactile'][np.newaxis, :, :], self.n_obs_steps, axis=0)
        else:
            tactile_ready = self.tactile_sensors.is_ready() if self.tactile_sensors is not None else False
            print(f"DEBUG: Tactile sensors not available - tactile_sensors: {self.tactile_sensors is not None}, is_ready: {tactile_ready}")
        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                obs_data,
                obs_align_timestamps,
            )
            
        obs_data['timestamp'] = obs_align_timestamps

        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())
        return obs_data


    def exec_actions(self,
                     joint_actions: np.ndarray,
                     eef_actions: np.ndarray,
                     timestamps: np.ndarray,
                     mode = 'joint', # 'joint' or 'eef'
                     dt: float = 0.1,
                     stages: Optional[np.ndarray] = None):
        """
        Execute a series of robot actions at specified times.

        Args:
            actions (np.ndarray): An array of actions to be executed by the robot. Each action
                                should correspond to a robot pose or similar set of instructions.
            timestamps (np.ndarray): An array of timestamps (in seconds since the epoch) when each 
                                    corresponding action should be executed. These should be future 
                                    times relative to the current time when exec_actions is called.
            stages (Optional[np.ndarray], optional): An optional array of stage identifiers for each 
                                                    action. Defaults to None. If provided, it helps 
                                                    in categorizing or identifying the stage of each action.

        Raises:
            AssertionError: If the method is called when the object is not ready to execute actions.

        Note:
            - The method filters out any actions and their corresponding timestamps and stages 
            if the timestamp is earlier than the current time.
            - Actions are scheduled as waypoints for the robot to follow.
            - If accumulators for actions or stages are set, the actions and stages are recorded.
        """
        # Ensure that the object is ready to execute actions
        assert self.is_ready

        # Convert input actions and timestamps to numpy arrays if they aren't already
        if not isinstance(joint_actions, np.ndarray):
            joint_actions = np.array(joint_actions)
        if not isinstance(eef_actions, np.ndarray):
            eef_actions = np.array(eef_actions) 
            
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
            
        # Initialize stages array if not provided, or convert to numpy array if needed
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64) # timestamps is dummy
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)        
            
        # real timestamp
        receive_time = time.time()
        is_new = timestamps > receive_time
        new_joint_actions = joint_actions[is_new]
        new_eef_actions = eef_actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        # schedule waypoints
        if mode == 'joint':
            for i in range(len(new_joint_actions)):
                self.robot.schedule_joints(new_joint_actions[i], new_timestamps[i])
        elif mode == 'eef':
            for i in range(len(new_eef_actions)):
                self.robot.schedule_waypoint(new_eef_actions[i], new_timestamps[i])

        # record actions
        # if self.action_accumulator is not None:
        #     self.action_accumulator.put(
        #         actions,
        #         timestamps
        #     )
        # record joint_actions
        if self.joint_action_accumulator is not None:
            self.joint_action_accumulator.put(
                new_joint_actions,
                new_timestamps
            )
        if self.eef_action_accumulator is not None:
            self.eef_action_accumulator.put(
                new_eef_actions,
                new_timestamps
            )

        if self.stage_accumulator is not None:
            self.stage_accumulator.put(
                stages,
                timestamps
            )

    def get_robot_state(self):
        if self.dummy_robot:
            return {'robot_eef_pose': np.zeros((6,), dtype=np.float32),
                    'TargetTCPPose': np.zeros((6,), dtype=np.float32),}
        else:
            return self.robot.get_state() 

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        hdf5_paths = list()
        zarr_paths = list()
        cam_ids = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))
            hdf5_paths.append(
                str(this_video_dir.joinpath(f'{i}.hdf5').absolute()))
            zarr_paths.append(
                str(this_video_dir.joinpath(f'{i}.zarr').absolute()))
            cam_ids.append(i)

        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(
            video_path=video_paths, 
            hdf5_path=hdf5_paths, 
            zarr_path=zarr_paths, 
            episode_id=episode_id, 
            cam_ids=cam_ids, 
            start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.joint_action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.eef_action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )

        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        if self.save_format == 'zarr':
            self.episode_id = self.replay_buffer.n_episodes
        print(f'Episode {self.episode_id} started!')

    def end_episode(self, incr_epi=True):
        "Stop recording"
        assert self.is_ready

        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            # assert self.action_accumulator is not None
            assert self.joint_action_accumulator is not None
            assert self.eef_action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps            

            num_cam = 0
            cam_width = -1
            cam_height = -1
            for key in obs_data.keys():
                if 'camera' in key and 'color' in key:
                    num_cam += 1
                    cam_height, cam_width = obs_data[key].shape[1:3]

            # actions = self.action_accumulator.actions
            # action_timestamps = self.action_accumulator.timestamps
            joint_actions = self.joint_action_accumulator.actions
            eef_actions = self.eef_action_accumulator.actions
            action_timestamps = self.joint_action_accumulator.timestamps
            
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            num_cam = self.realsense.n_cameras
            if n_steps > 0:
                ### init episode data
                episode = {
                    'timestamp': None,
                    'stage': None,
                    'observations': 
                        {'robot_joint': [],
                         'robot_joint_vel': [], 
                         'robot_base_pose_in_world': [],
                         'robot_eef_pose': [],
                         'robot_eef_pose_vel': [],
                        #  'robot_gripper_pos': {},
                         'images': {},},
                    'joint_action': [],
                    'cartesian_action': [],
                }
                for cam in range(num_cam):
                    episode['observations']['images'][f'camera_{cam}_color'] = []
                    episode['observations']['images'][f'camera_{cam}_depth'] = []
                    episode['observations']['images'][f'camera_{cam}_intrinsics'] = []
                    episode['observations']['images'][f'camera_{cam}_extrinsics'] = []

                attr_dict = {
                }

                ### create config dict
                config_dict = {
                    'observations': {
                        'images': {}
                    },
                    'timestamp': {
                        'dtype': 'float64'
                    },
                }
                
                for cam in range(num_cam):
                    color_save_kwargs = {
                        'chunks': (1, cam_height, cam_width, 3), # (1, 480, 640, 3)
                        'compression': 'gzip',
                        'compression_opts': 9,
                        'dtype': 'uint8',
                    }
                    depth_save_kwargs = {
                        'chunks': (1, cam_height, cam_width), # (1, 480, 640)
                        'compression': 'gzip',
                        'compression_opts': 9,
                        'dtype': 'uint16',
                    }
                    config_dict['observations']['images'][f'camera_{cam}_color'] = color_save_kwargs
                    config_dict['observations']['images'][f'camera_{cam}_depth'] = depth_save_kwargs
               
                ### load episode data
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['joint_action'] = joint_actions[:n_steps]
                episode['cartesian_action'] = eef_actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                # for key, value in obs_data.items():
                #     episode[key] = value[:n_steps]
                for key, value in obs_data.items():
                    if 'camera' in key:
                        episode['observations']['images'][key] = value[:n_steps]
                    else:
                        episode['observations'][key] = value[:n_steps]
                if self.save_format == 'zarr':
                    self.replay_buffer.add_episode(episode, compressors='disk')
                    episode_id = self.replay_buffer.n_episodes - 1
                    # assert episode_id == self.episode_id
                    # self.episode_id = episode_id 
                    print(f'Episode {episode_id} saved!')

                else:
                    episode_path = self.output_dir.joinpath(f'episode_{self.episode_id}.hdf5')
                    save_dict_to_hdf5(episode, config_dict, str(episode_path), attr_dict=attr_dict)

                    print(f'Episode {self.episode_id} saved!')
                    if incr_epi:
                        self.episode_id += 1

            self.obs_accumulator = None
            self.joint_action_accumulator = None
            self.eef_action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        if self.save_format == 'zarr':
            self.end_episode()
            if self.replay_buffer.n_episodes == 0:
                print("No episode to drop!")
                return
            self.replay_buffer.drop_episode()
            episode_id = self.replay_buffer.n_episodes 
            # self.episode_id = episode_id
            this_video_dir = self.video_dir.joinpath(str(episode_id))
            if this_video_dir.exists():
                shutil.rmtree(str(this_video_dir))
            print(f'Episode {episode_id} dropped!')
        else:
            self.realsense.stop_recording()
            self.obs_accumulator = None
            self.joint_action_accumulator = None
            self.eef_action_accumulator = None

            self.stage_accumulator = None
            
            this_video_dir = self.video_dir.joinpath(str(self.episode_id))
            if this_video_dir.exists():
                shutil.rmtree(str(this_video_dir))
            print(f'Episode {self.episode_id} dropped!')



# def test_episode_start():
#     # create env
#     os.system('mkdir -p tmp')
#     with RealUR5eEnv(
#             output_dir='tmp',
#         ) as env:
#         print('Created env!')
        
#         env.start_episode()
#         print('Started episode!')

# def test_env_obs_latency():
#     os.system('mkdir -p tmp')
#     with RealUR5eEnv(
#             output_dir='tmp',
#         ) as env:
#         print('Created env!')

#         for i in range(100):
#             start_time = time.time()
#             obs = env.get_obs()
#             end_time = time.time()
#             print(f'obs latency: {end_time - start_time}')
#             time.sleep(0.1)

# def test_env_demo_replay():
#     ctrl_mode = 'joint'
#     os.system('mkdir -p tmp')
#     demo_path = '/home/haonan/Projects/bimanual_ur5e/data/real_laundry/episode_0.hdf5'
#     demo_dict, _ = load_dict_from_hdf5(demo_path)
#     if ctrl_mode == 'joint':
#         actions = demo_dict['joint_action']
#     else:
#         actions = demo_dict['cartesian_action']
#     with RealUR5eEnv(
#             output_dir='tmp',
#             ctrl_mode=ctrl_mode,
#         ) as env:
#         print('Created env!')

#         timestamps = time.time() + np.arange(len(actions)) / 10 + 1.0
#         start_step = 0
#         if ctrl_mode == 'joint':
#             env.robot.set_robot_joints(actions[0])

#         while True:
#             curr_time = time.monotonic()
#             loop_end_time = curr_time + 1.0
#             end_step = min(start_step+10, len(actions))
#             action_batch = actions[start_step:end_step]
#             timestamp_batch = timestamps[start_step:end_step]
#             if ctrl_mode == 'joint':
#                 env.exec_actions(
#                     joint_actions=action_batch,
#                     eef_actions=np.zeros((action_batch.shape[0], 7)),
#                     timestamps=timestamp_batch,
#                     mode=ctrl_mode,
#                 )
#             else:
#                 env.exec_actions(
#                     joint_actions=np.zeros((action_batch.shape[0], 7)),
#                     eef_actions=action_batch,
#                     timestamps=timestamp_batch,
#                     mode=ctrl_mode,
#                 )
#             print(f'executed {end_step - start_step} actions')
#             start_step = end_step
#             precise_wait(loop_end_time)
#             if start_step >= len(actions):
#                 break

def test_env_demo_replay(ctrl_mode='eef', episode_num=0, no_filter=True):
    """Replay puzzle expert demonstrations on real robot"""
    d = dirname(dirname(dirname(abspath(__file__))))
    zarr_path = os.path.join(d, 'data/puzzle_expert_30.zarr')
    
    print(f"Loading puzzle expert dataset from: {zarr_path}")
    assert os.path.exists(zarr_path), f"Dataset not found at {zarr_path}"
    
    # Load dataset
    group = zarr.open_group(zarr_path, 'r')
    demo_dict = group['data']
    episode_ends = group['meta']['episode_ends'][:]
    
    print(f"Dataset loaded - Episodes available: {len(episode_ends)}")
    print(f"Episode ends: {episode_ends}")
    
    # Get episode boundaries
    if episode_num == 0:
        episode_start = 0
        episode_end = episode_ends[0]
    else:
        episode_start = episode_ends[episode_num-1]
        episode_end = episode_ends[episode_num]
    
    episode_slice = slice(episode_start, episode_end)
    print(f"Replaying episode {episode_num}: timesteps {episode_start} to {episode_end-1}")
    
    # Load actions based on control mode
    if ctrl_mode == 'joint':
        # Use recorded joint positions as target actions
        actions = demo_dict['robot_joint'][episode_slice]
        print(f"Using joint actions - Shape: {actions.shape}")
    else:  # eef mode
        # Use recorded EEF poses as target actions
        actions = demo_dict['robot_eef_pose'][episode_slice]
        print(f"Using EEF poses - Shape: {actions.shape}")
        
        # Debug rotation data format
        print(f"Raw rotation data shape: {actions[:,3:6].shape}")
        print(f"Sample rotation data: {actions[0,3:6]}")
        
        # The data appears to already be in axis-angle format, so no transformation needed
        print("Skipping rotation transformation - data is already in axis-angle format")
    
    # Check for tactile data
    has_tactile = 'left_tactile' in demo_dict and 'right_tactile' in demo_dict
    if has_tactile:
        left_tactile = demo_dict['left_tactile'][episode_slice]
        right_tactile = demo_dict['right_tactile'][episode_slice]
        print(f"Tactile data found - Left: {left_tactile.shape}, Right: {right_tactile.shape}")
    
    # Filter out non-moving segments for smoother replay
    if not no_filter:
        if ctrl_mode == 'joint':
            # Filter based on joint movement
            # Handle gripper dimension mismatch - only compare first 6 joints
            obs_joints = demo_dict['robot_joint'][episode_slice]
            joint_actions = demo_dict['joint_action'][episode_slice]
            
            # Make sure dimensions match for comparison
            obs_joints_6d = obs_joints[:, :6] if obs_joints.shape[1] > 6 else obs_joints
            joint_actions_6d = joint_actions[:, :6] if joint_actions.shape[1] > 6 else joint_actions
            
            not_moving_mask = are_joints_close(obs_joints_6d, joint_actions_6d)
            moving_mask = ~not_moving_mask
            actions_filtered = actions[moving_mask]
        else:
            # Filter based on pose differences
            pose_diff = np.diff(actions[:, :3], axis=0)
            pose_movement = np.linalg.norm(pose_diff, axis=1)
            movement_threshold = 0.001  # 1mm threshold
            moving_indices = np.where(pose_movement > movement_threshold)[0] + 1
            moving_mask = np.zeros(len(actions), dtype=bool)
            moving_mask[0] = True  # Always include first pose
            moving_mask[moving_indices] = True
            actions_filtered = actions[moving_mask]
        
        print(f"Filtered actions - Original: {len(actions)}, Moving: {len(actions_filtered)}")
    else:
        actions_filtered = actions
        print(f"No filtering applied - Using all {len(actions)} actions")
    
    # Create environment and replay (corrected for single-arm puzzle task)
    with RealUR5eEnv(
            output_dir='tmp_replay',
            ctrl_mode=ctrl_mode,
            speed_slider_value=0.05,  # Slower for safety
            single_arm_type='right',  # Assuming right arm for puzzle
            use_gripper=True,
            tactile_sensors=None  # No tactile sensors for replay
        ) as env:
        print('Environment created, starting replay...')
        
        # Generate timestamps at 10 Hz
        timestamps = time.time() + np.arange(len(actions_filtered)) / 10 + 2.0
        start_step = 0
        
        print(f"Replaying {len(actions_filtered)} actions...")
        
        while start_step < len(actions_filtered):
            curr_time = time.monotonic()
            loop_end_time = curr_time + 1.0
            
            # Execute actions in batches of 10
            end_step = min(start_step + 10, len(actions_filtered))
            action_batch = actions_filtered[start_step:end_step]
            timestamp_batch = timestamps[start_step:end_step]
            
            if ctrl_mode == 'joint':
                # Joint control mode
                env.exec_actions(
                    joint_actions=action_batch,
                    eef_actions=np.zeros((action_batch.shape[0], 7)),
                    timestamps=timestamp_batch,
                    mode=ctrl_mode,
                )
            else:
                # EEF control mode
                env.exec_actions(
                    joint_actions=np.zeros((action_batch.shape[0], 6)),
                    eef_actions=action_batch,
                    timestamps=timestamp_batch,
                    mode=ctrl_mode,
                )
            
            print(f'Executed actions {start_step} to {end_step-1} / {len(actions_filtered)}')
            start_step = end_step
            
            # Wait for next batch
            precise_wait(loop_end_time)
        
        print("Replay completed successfully!")

# def test_cache_replay():
#     import zarr
#     import pytorch3d.transforms
#     import torch
#     import scipy.spatial.transform as st
#     ctrl_mode = 'eef'

#     os.system('mkdir -p tmp')
    
#     # only to get initial joint pos
#     demo_path = '/home/haonan/Projects/bimanual_ur5e/data/real_cake_serving/episode_0.hdf5'
#     demo_dict, _ = load_dict_from_hdf5(demo_path)

#     cache_path = '/home/haonan/Projects/bimanual_ur5e/data/real_cake_serving/5a116216bc1207fe82c7f1a4771b60f2.zarr.zip'
#     with zarr.ZipStore(cache_path, mode='r') as zip_store:
#         replay_buffer = ReplayBuffer.copy_from_store(
#             src_store=zip_store, store=zarr.MemoryStore())
#     actions = replay_buffer['action']
#     with RealUR5eEnv(
#             output_dir='tmp',
#             ctrl_mode=ctrl_mode,
#         ) as env:
#         print('Created env!')

#         timestamps = time.time() + np.arange(len(actions)) / 10 + 1.0
#         # ik_init = [demo_dict['observations']['full_joint_pos'][0]] * len(actions)
#         # print(demo_dict['observations']['full_joint_pos'][()])

#         # convert action from rotation 6d to euler
#         actions = np.asarray(actions)
#         actions_reshape = actions.reshape(actions.shape[0], 10)
#         action_pos = actions_reshape[:,:3]
#         action_rot_6d = actions_reshape[:,3:9]
#         action_rot_mat = pytorch3d.transforms.rotation_6d_to_matrix(torch.from_numpy(action_rot_6d)).numpy()
#         action_rot_euler = st.Rotation.from_matrix(action_rot_mat).as_rotvec()
#         actions_reshape = np.concatenate([action_pos, action_rot_euler, actions_reshape[:,-1:]], axis=-1)
#         actions = actions_reshape.reshape(actions.shape[0], 7)

#         start_step = 0
#         while True:
#             curr_time = time.monotonic()
#             loop_end_time = curr_time + 1.0
#             end_step = min(start_step+10, len(actions))
#             action_batch = actions[start_step:end_step]
#             timestamp_batch = timestamps[start_step:end_step]
#             # ik_init_batch = ik_init[start_step:end_step]
#             if ctrl_mode == 'joint':
#                 env.exec_actions(
#                     joint_actions=action_batch,
#                     eef_actions=np.zeros((action_batch.shape[0], 7)),
#                     timestamps=timestamp_batch,
#                     mode=ctrl_mode,
#                 )
#             else:
#                 env.exec_actions(
#                     joint_actions=np.zeros((action_batch.shape[0], 7)),
#                     eef_actions=action_batch,
#                     timestamps=timestamp_batch,
#                     mode=ctrl_mode,
#                 )
#             print(f'executed {end_step - start_step} actions')
#             start_step = end_step
#             precise_wait(loop_end_time)
#             if start_step >= len(actions):
#                 break

if __name__ == '__main__':
    # Dataset replay functionality has been moved to scripts/replay_dataset.py
    # Use: python scripts/replay_dataset.py data/puzzle_expert_30.zarr
    print("Dataset replay functionality has been moved to scripts/replay_dataset.py")
    print("Usage: python scripts/replay_dataset.py data/puzzle_expert_30.zarr")
    print("See real_world.md for detailed instructions.")
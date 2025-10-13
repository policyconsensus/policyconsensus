import pathlib
import numpy as np
import time
import shutil
import math
from multiprocessing.managers import SharedMemoryManager
from modular_policy.real_world.multi_realsense import MultiRealsense, SingleRealsense
from modular_policy.real_world.video_recorder import VideoRecorder
from modular_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
)
from modular_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from modular_policy.common.replay_buffer import ReplayBuffer
from modular_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)
import cProfile
import pstats
import io
from modular_policy.common.cv2_util import get_image_transform


class RealsenseEnv:
    def __init__(self,
                 # required params
                 output_dir,
                 # env params
                 frequency=10,
                 n_obs_steps=2,
                 # obs
                 obs_image_resolution=(640, 480),
                 max_obs_buffer_size=30,
                 camera_serial_numbers=None,
                 obs_float32=False,
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

        self.episode_id = 0

        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')
        self.debug = debug
        self.enable_depth = enable_depth

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
            
        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec='h264',
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video,
            video_capture_resolution=video_capture_resolution,
            num_cams=len(camera_serial_numbers),
        )

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
            vis_transform=vis_transform if enable_multi_cam_vis else None, # TODO: it is blocking the program
            recording_transform=recording_transfrom,
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

        self.realsense = realsense
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        # recording
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        self.episode_id = replay_buffer.n_episodes
        # temp memory buffers
        self.last_realsense_data = None
        # recording buffers
        self.obs_accumulator = None
        self.stage_accumulator = None

        self.start_time = None
        
        
    # ======== start-stop API =============
    @property
    def is_ready(self):
        return self.realsense.is_ready 

    def start(self, wait=True):
        if self.debug:
            print("Starting RealPushLEnv...")
        self.realsense.start(wait=False)
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
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
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
    def get_obs(self, stages=None, profile=False) -> dict:
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


        # return obs
        obs_data = dict(camera_obs)
        
        # accumulate obs
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                obs_data,
                obs_align_timestamps,
            )
        if self.stage_accumulator is not None:
            if stages is None:
                stages = np.zeros_like(obs_align_timestamps, dtype=np.int64)

            self.stage_accumulator.put(
                stages,
                obs_align_timestamps
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

        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.episode_id = self.replay_buffer.n_episodes
        print(f'Episode {self.episode_id} started!')

    def end_episode(self, incr_epi=True):
        "Stop recording"
        assert self.is_ready

        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
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
            
            stages = self.stage_accumulator.actions
            n_steps = len(obs_timestamps)
            num_cam = self.realsense.n_cameras
            if n_steps > 0:
                ### init episode data
                episode = {
                    'timestamp': None,
                    'stage': None,
                    'observations': 
                        {'images': {},},
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
                episode['stage'] = stages[:n_steps]
                # for key, value in obs_data.items():
                #     episode[key] = value[:n_steps]
                for key, value in obs_data.items():
                    if 'camera' in key:
                        episode['observations']['images'][key] = value[:n_steps]
                    else:
                        episode['observations'][key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')


            self.obs_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
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

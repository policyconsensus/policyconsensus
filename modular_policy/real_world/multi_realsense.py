from typing import List, Optional, Union, Dict, Callable
from collections import OrderedDict
import numbers
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from modular_policy.real_world.single_realsense import SingleRealsense
from modular_policy.real_world.video_recorder import VideoRecorder


class MultiRealsense:
    def __init__(self,
        serial_numbers: Optional[List[str]]=None,
        shm_manager: Optional[SharedMemoryManager]=None,
        resolution=(1280,720),
        capture_fps=30,
        put_fps=None,
        put_downsample=True,
        record_fps=None,
        enable_color=True,
        enable_depth=True,
        enable_infrared=False,
        enable_pointcloud=False,
        process_pcd=False,
        draw_keypoints=False,
        keypoint_kwargs=None,
        get_max_k=30,
        advanced_mode_config: Optional[Union[dict, List[dict]]]=None,
        transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        vis_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        recording_transform: Optional[Union[Callable[[Dict], Dict], List[Callable]]]=None,
        video_recorder: Optional[Union[VideoRecorder, List[VideoRecorder]]]=None,
        verbose=False
        ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        n_cameras = len(serial_numbers)

        advanced_mode_config = repeat_to_list(
            advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(
            transform, n_cameras, Callable)
        vis_transform = repeat_to_list(
            vis_transform, n_cameras, Callable)
        recording_transform = repeat_to_list(
            recording_transform, n_cameras, Callable)

        video_recorders = repeat_to_list(
            video_recorder, n_cameras, VideoRecorder)
        
        # self.extrinsics = load_extrinsic_matrices(os.path.join(os.path.dirname(__file__), "extrinsics.json"))

        cameras = OrderedDict()
        for i, serial in enumerate(serial_numbers):
            cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                record_fps=record_fps,
                enable_color=enable_color,
                enable_depth=enable_depth,
                enable_infrared=enable_infrared,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                vis_transform=vis_transform[i],
                recording_transform=recording_transform[i],
                video_recorder=video_recorders[i],
                verbose=verbose,
                # extrinsics=self.extrinsics[serial] if serial in self.extrinsics else None
            )
        self.cameras = cameras
        self.shm_manager = shm_manager
        
        self.enable_pointcloud = enable_pointcloud
        self.process_pcd = process_pcd
        self.draw_keypoints = draw_keypoints
        self.keypoint_kwargs = keypoint_kwargs
        
        self.transform = transform
        
        self.prev_trans_local_world = [None]*2
        self.prev_trans_local_world_vis = [None]*2
        

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    @property
    def n_cameras(self):
        return len(self.cameras)
    
    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready
    
    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)
        
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)
        
        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()
    
    def get(self, k=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out


        return out

    def get_vis(self, out=None):
        results = list()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if out is not None:
                this_out = dict()
                for key, v in out.items():
                    # use the slicing trick to maintain the array
                    # when v is 1D
                    this_out[key] = v[i:i+1].reshape(v.shape[1:])
            this_out = camera.get_vis(out=this_out)
            if out is None:
                results.append(this_out)
        if out is None:
            out = dict()
            for key in results[0].keys():
                out[key] = np.stack([x[key] for x in results])
                
        if self.enable_pointcloud:
            boundaries = {'x_lower': -0.15, 'x_upper': 0.38, 'y_lower': -0.55, 'y_upper': 0.125, 'z_lower': -0.02, 'z_upper': 0.07}

            # shape: (n_cameras, H, W, C)
            colors = out['color'][..., ::-1] # please note that the color should be in RGB
            depths = out['depth'] 
            intrinsics = out['intrinsics']
            extrinsics = out['extrinsics']
            dist_coeffs = out['dist_coeffs']
            
            raise RuntimeError("Not implemented `extract_keypoint_from_img`")
            keypoints_world, all_trans_local_world = extract_keypoint_from_img(colors, depths, intrinsics, extrinsics, boundaries, self.prev_trans_local_world_vis, **self.keypoint_kwargs)
            self.prev_trans_local_world_vis = all_trans_local_world
            keypoints_world = np.concatenate(keypoints_world, axis=0)
            if self.draw_keypoints:
                for camera_idx in range(self.n_cameras):
                    draw_keypoints_on_image(camera_idx, keypoints_world, colors, intrinsics, extrinsics, dist_coeffs)


        return out
    
    def set_color_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_color_option(option, value[i])


    def set_depth_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_option(option, value[i])

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        n_camera = len(self.cameras)
        exposure= repeat_to_list(exposure, n_camera, numbers.Number)
        gain= repeat_to_list(gain, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_exposure(exposure[i], gain[i])


    def set_depth_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """
        n_camera = len(self.cameras)
        exposure= repeat_to_list(exposure, n_camera, numbers.Number)
        gain= repeat_to_list(gain, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_exposure(exposure[i], gain[i])

    def set_depth_preset(self, preset):
        n_camera = len(self.cameras)
        preset = repeat_to_list(preset, n_camera, str)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_depth_preset(preset[i])

    def set_white_balance(self, white_balance=None):
        n_camera = len(self.cameras)
        white_balance= repeat_to_list(white_balance, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_white_balance(white_balance[i])

    def set_contrast(self, contrast=None):
        n_camera = len(self.cameras)
        contrast= repeat_to_list(contrast, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_contrast(contrast[i])

    def get_intr_mat(self):
        return np.array([c.get_intr_mat() for c in self.cameras.values()])
    
    def get_depth_scale(self):
        return np.array([c.get_depth_scale() for c in self.cameras.values()])
    
    def start_recording(self, video_path: Union[str, List[str]], 
                        hdf5_path: Union[str, List[str]], 
                        zarr_path: Union[str, List[str]],
                        episode_id: int, 
                        cam_ids: Union[str, List[int]],
                        start_time: float):
        if isinstance(video_path, str):
            # directory
            video_dir = pathlib.Path(video_path)
            assert video_dir.parent.is_dir()
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = list()
            for i in range(self.n_cameras):
                video_path.append(
                    str(video_dir.joinpath(f'{i}.mp4').absolute()))
        assert len(video_path) == self.n_cameras

        if isinstance(hdf5_path, str):
            # directory
            hdf5_dir = pathlib.Path(hdf5_path)
            assert hdf5_dir.parent.is_dir()
            hdf5_dir.mkdir(parents=True, exist_ok=True)
            hdf5_path = list()
            for i in range(self.n_cameras):
                hdf5_path.append(
                    str(hdf5_dir.joinpath(f'{i}.hdf5').absolute()))
                
        if isinstance(zarr_path, str):
            # directory
            zarr_dir = pathlib.Path(zarr_path)
            assert zarr_dir.parent.is_dir()
            zarr_dir.mkdir(parents=True, exist_ok=True)
            zarr_path = list()
            for i in range(self.n_cameras):
                zarr_path.append(
                    str(zarr_dir.joinpath(f'{i}.zarr').absolute()))

        for i, camera in enumerate(self.cameras.values()):
            camera.start_recording(video_path[i], 
                                   hdf5_path[i] if hdf5_path is not None else None, 
                                   zarr_path[i] if zarr_path is not None else None,
                                   episode_id, 
                                   cam_ids[i],
                                   start_time)
    
    def stop_recording(self):
        for i, camera in enumerate(self.cameras.values()):
            camera.stop_recording()
    
    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)


def repeat_to_list(x, n: int, cls):  
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x
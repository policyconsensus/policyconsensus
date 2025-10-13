from typing import Optional, Callable, Dict
from multiprocessing import Manager
import os
import enum
import time
import warnings
import json
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
import cv2
from threadpoolctl import threadpool_limits
from multiprocessing.managers import SharedMemoryManager
import matplotlib.pyplot as plt
import sys
from modular_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
from modular_policy.shared_memory.shared_ndarray import SharedNDArray
from modular_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from modular_policy.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from modular_policy.real_world.video_recorder import VideoRecorder



class Command(enum.Enum):
    SET_COLOR_OPTION = 0
    SET_DEPTH_OPTION = 1
    START_RECORDING = 2
    STOP_RECORDING = 3
    RESTART_PUT = 4

class SingleRealsense(mp.Process):
    MAX_PATH_LENGTH = 4096 # linux path has a limit of 4096 bytes

    def __init__(
            self, 
            shm_manager: SharedMemoryManager,
            serial_number,
            resolution=(1280,720),
            capture_fps=30,
            put_fps=None,
            put_downsample=True,
            record_fps=None,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=30,
            advanced_mode_config=None,
            transform: Optional[Callable[[Dict], Dict]] = None,
            vis_transform: Optional[Callable[[Dict], Dict]] = None,
            recording_transform: Optional[Callable[[Dict], Dict]] = None,
            video_recorder: Optional[VideoRecorder] = None,
            verbose=False,
        ):
        super().__init__()

        if put_fps is None:
            put_fps = capture_fps
        if record_fps is None:
            record_fps = capture_fps

        # create ring buffer
        resolution = tuple(resolution)
        shape = resolution[::-1]
        examples = dict()
        if enable_color:
            examples['color'] = np.empty(
                shape=shape+(3,), dtype=np.uint8)
        if enable_depth:
            examples['depth'] = np.empty(
                shape=np.array(shape).astype(int), dtype=np.uint16)

        if enable_infrared:
            examples['infrared'] = np.empty(
                shape=shape, dtype=np.uint8)
        examples['camera_capture_timestamp'] = 0.0
        examples['camera_receive_timestamp'] = 0.0
        examples['timestamp'] = 0.0
        examples['step_idx'] = 0

        vis_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if vis_transform is None 
                else vis_transform(dict(examples)),
            get_max_k=1,
            get_time_budget=0.2,
            put_desired_frequency=capture_fps
        )

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples if transform is None
                else transform(dict(examples)),
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=put_fps
        )

        # create command queue
        examples = {
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': rs.option.exposure.value,
            'option_value': 0.0,
            'video_path': np.array('a'*self.MAX_PATH_LENGTH),
            'hdf5_path': np.array('a'*self.MAX_PATH_LENGTH),
            'zarr_path': np.array('a'*self.MAX_PATH_LENGTH),
            'episode_id': 0,
            'cam_id': 0,
            'recording_start_time': 0.0,
            'put_start_time': 0.0
        }

        command_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=examples,
            buffer_size=128
        )

        # create shared array for intrinsics
        intrinsics_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(7,),
                dtype=np.float64)
        intrinsics_array.get()[:] = 0

        dist_coeff_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(5,),
                dtype=np.float64)
        dist_coeff_array.get()[:] = 0

        # create video recorder
        if video_recorder is None:
            # realsense uses bgr24 pixel format
            # default thread_type to FRAEM
            # i.e. each frame uses one core
            # instead of all cores working on all frames.
            # this prevents CPU over-subpscription and
            # improves performance significantly
            video_recorder = VideoRecorder.create_h264(
                fps=record_fps, 
                codec='h264',
                input_pix_fmt='bgr24', 
                crf=18,
                thread_type='FRAME',
                thread_count=1)

        # copied variables
        self.serial_number = serial_number
        self.resolution = resolution
        self.capture_fps = capture_fps
        self.put_fps = put_fps
        self.put_downsample = put_downsample
        self.record_fps = record_fps
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_infrared = enable_infrared
        self.advanced_mode_config = advanced_mode_config
        self.transform = transform
        self.vis_transform = vis_transform
        self.recording_transform = recording_transform
        self.video_recorder = video_recorder
        self.verbose = verbose
        self.put_start_time = None

        # shared variables
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.ring_buffer = ring_buffer
        self.vis_ring_buffer = vis_ring_buffer
        self.command_queue = command_queue
        self.intrinsics_array = intrinsics_array
        self.dist_coeff_array = dist_coeff_array

        manager = Manager()
        self.intrinsics_dict = manager.dict()
    
    @staticmethod
    def get_connected_devices_serial():
        serials = list()
        for d in rs.context().devices:
            if d.get_info(rs.camera_info.name).lower() != 'platform camera':
                serial = d.get_info(rs.camera_info.serial_number)
                product_line = d.get_info(rs.camera_info.product_line)
                if product_line == 'D400' : # or product_line == 'SR300':
                    # only works with D400 series
                    serials.append(serial)
        serials = sorted(serials)
        return serials

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= user API ===========
    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.end_wait()

    def start_wait(self):
        self.ready_event.wait()
    
    def end_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def get(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k, out=out)
    
    def get_vis(self, out=None):
        return self.vis_ring_buffer.get(out=out)
    
    # ========= user API ===========
    def set_color_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_COLOR_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })
        
    def set_depth_option(self, option: rs.option, value: float):
        self.command_queue.put({
            'cmd': Command.SET_DEPTH_OPTION.value,
            'option_enum': option.value,
            'option_value': value
        })

    def set_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)
                
    def set_depth_exposure(self, exposure=None, gain=None):
        """
        exposure: (1, 10000) 100us unit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_depth_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_depth_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_depth_option(rs.option.gain, gain)

    def set_depth_preset(self, preset: str):
        visual_preset = {
            'Custom': 0,
            'Default': 1,
            'Hand': 2,
            'High Accuracy': 3,
            'High Density': 4,
        }
        self.set_depth_option(rs.option.visual_preset, visual_preset[preset])


    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def set_contrast(self, contrast=None):
        if contrast is None:
            self.set_color_option(rs.option.contrast, 0)
        else:
            self.set_color_option(rs.option.contrast, contrast)

    def get_intr_mat(self):
        assert self.ready_event.is_set()
        fx, fy, ppx, ppy = self.intrinsics_array.get()[:4]
        mat = np.eye(3)
        mat[0,0] = fx
        mat[1,1] = fy
        mat[0,2] = ppx
        mat[1,2] = ppy
        return mat

    def get_dist_coeff(self):
        assert self.ready_event.is_set()
        return np.array(self.dist_coeff_array.get()[:])

    def get_intrinsics(self, pipeline_profile, stream_type):
        """
        Retrieves intrinsic parameters for a given stream.
        """
        stream = pipeline_profile.get_stream(stream_type)
        intr = stream.as_video_stream_profile().get_intrinsics()
        return {
            'fx': intr.fx,
            'fy': intr.fy,
            'ppx': intr.ppx,
            'ppy': intr.ppy,
            'height': intr.height,
            'width': intr.width,
            'coeffs': intr.coeffs,
        }
        
    def save_intrinsics(self, pipeline_profile):
        """
        Saves the intrinsic parameters for color and depth streams.
        """
        self.intrinsics_dict['color'] = self.get_intrinsics(pipeline_profile, rs.stream.color)

        if self.enable_depth:
            self.intrinsics_dict['depth'] = self.get_intrinsics(pipeline_profile, rs.stream.depth)
            depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            self.intrinsics_dict['depth']['scale'] = depth_scale
            # if depth_sensor.supports(rs.option.emitter_enabled):
            #     depth_sensor.set_option(rs.option.emitter_enabled, 1)
            #     depth_sensor.set_option(rs.option.laser_power, 330)

            # depth2color = pipeline_profile.get_stream(rs.stream.depth).as_video_stream_profile().get_extrinsics_to(pipeline_profile.get_stream(rs.stream.color))
            # self.depth2color_array.get()[:3, :3] = np.array(depth2color.rotation).reshape(3,3)
            # self.depth2color_array.get()[:3, 3] = np.array(depth2color.translation).reshape(3)

        self.dist_coeff_array.get()[:] = np.array(self.intrinsics_dict['color']['coeffs'])

    def get_depth_scale(self):
        assert self.ready_event.is_set()
        scale = self.intrinsics_array.get()[-1]
        return scale
    
    def start_recording(self, video_path: str, hdf5_path: str, zarr_path: str, episode_id: int, cam_id: int, start_time: float=-1):
        assert self.enable_color

        path_len = len(video_path.encode('utf-8'))
        if path_len > self.MAX_PATH_LENGTH:
            raise RuntimeError('video_path too long.')
        self.command_queue.put({
            'cmd': Command.START_RECORDING.value,
            'video_path': video_path,
            'hdf5_path': hdf5_path,
            'zarr_path': zarr_path,
            'cam_id': cam_id,
            'episode_id': episode_id,
            'recording_start_time': start_time
        })
        
    def stop_recording(self):
        self.command_queue.put({
            'cmd': Command.STOP_RECORDING.value
        })
    
    def restart_put(self, start_time):
        self.command_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'put_start_time': start_time
        })

    def _init_depth_process(self):
        # Initialize the processing steps
        self.depth_to_disparity = rs.disparity_transform(True)
        self.disparity_to_depth = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        # self.spatial.set_option(rs.option.holes_fill, 1)
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 1)
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, 2)
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        # Initialize the alignment to make the depth data aligned to the rgb camera coordinate
        # self.align = rs.align(rs.stream.color)

    def project_point_to_pixel(self, points):
        # The points here should be in the camera coordinate, n*3
        points = np.array(points)
        pixels = []
        # # Use the realsense projeciton, however, it's slow for the loop; this can give nan for invalid points
        # for i in range(len(points)):
        #     pixels.append(rs.rs2_project_point_to_pixel(self.intrinsic, points))
        # pixels = np.array(pixels)

        # Use the opencv projection
        # The width and height are inversed here
        pixels = cv2.projectPoints(
            points,
            np.zeros(3),
            np.zeros(3),
            self.intrinsic_matrix,
            self.dist_coef,
        )[0][:, 0, :]

        return pixels[:, ::-1]

    def deproject_pixel_to_point(self, pixel_depth):
        # pixel_depth contains [i, j, depth[i, j]]
        points = []
        for i in range(len(pixel_depth)):
            # The width and height are inversed here
            points.append(
                rs.rs2_deproject_pixel_to_point(
                    self.intrinsic,
                    [pixel_depth[i, 1], pixel_depth[i, 0]],
                    pixel_depth[i, 2],
                )
            )
        return np.array(points)

    def _process_depth(self, depth_frame):
        # Depth process
        filtered_depth = self.depth_to_disparity.process(depth_frame)
        filtered_depth = self.spatial.process(filtered_depth)
        filtered_depth = self.temporal.process(filtered_depth)
        filtered_depth = self.disparity_to_depth.process(filtered_depth)
        return filtered_depth

    # ========= interval API ===========
    def run(self):
        # limit threads
        threadpool_limits(1)
        cv2.setNumThreads(1)

        w, h = self.resolution
        fps = self.capture_fps
        # align = rs.align(rs.stream.color)
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        if self.enable_color:
            rs_config.enable_stream(rs.stream.color, 
                w, h, rs.format.bgr8, fps)
        if self.enable_depth:
            rs_config.enable_stream(rs.stream.depth,
                w, h, rs.format.z16, fps)
            self._init_depth_process()

        if self.enable_infrared:
            rs_config.enable_stream(rs.stream.infrared,
                w, h, rs.format.y8, fps)
        
        try:
            rs_config.enable_device(self.serial_number)

            # start pipeline
            pipeline = rs.pipeline()
            pipeline_profile = pipeline.start(rs_config)
            
            # do the align operation after the pipeline has started 
            # https://github.com/IntelRealSense/librealsense/issues/4224
            # https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
            align = rs.align(rs.stream.color)

            # report global time
            # https://github.com/IntelRealSense/librealsense/pull/3909
            d = pipeline_profile.get_device().first_color_sensor()
            d.set_option(rs.option.global_time_enabled, True)

            # setup advanced mode
            if self.advanced_mode_config is not None:
                json_text = json.dumps(self.advanced_mode_config)
                device = pipeline_profile.get_device()
                advanced_mode = rs.rs400_advanced_mode(device)
                advanced_mode.load_json(json_text)

            # get
            # color_stream = pipeline_profile.get_stream(rs.stream.color)
            # intr = color_stream.as_video_stream_profile().get_intrinsics()
            self.save_intrinsics(pipeline_profile)
            order = ['fx', 'fy', 'ppx', 'ppy', 'height', 'width']
            for i, name in enumerate(order):
                self.intrinsics_array.get()[i] = self.intrinsics_dict['color'].get(name) # getattr(self.intrinsics['color'], name)


            # if self.enable_depth:
            #     depth_sensor = pipeline_profile.get_device().first_depth_sensor()
            #     depth_scale = depth_sensor.get_depth_scale()
            #     self.intrinsics_array.get()[-1] = depth_scale
                
            #     depth_steam = pipeline_profile.get_stream(rs.stream.depth)
            #     depth_intr = depth_steam.as_video_stream_profile().get_intrinsics()

            # one-time setup (intrinsics etc, ignore for now)
            if self.verbose:
                print(f'[SingleRealsense {self.serial_number}] Main loop started.')

            # put frequency regulation
            put_idx = None
            put_start_time = self.put_start_time
            if put_start_time is None:
                put_start_time = time.time()

            iter_idx = 0
            t_start = time.time()
            cam_mat_saved = False
            while not self.stop_event.is_set():
                wait_start_time = time.time()
                # wait for frames to come in
                frameset = pipeline.wait_for_frames()
                receive_time = time.time()
                # align frames to color
                frameset = align.process(frameset)
                wait_time = time.time() - wait_start_time

                # grab data
                grab_start_time = time.time()
                data = dict()
                data['camera_receive_timestamp'] = receive_time
                # realsense report in ms
                data['camera_capture_timestamp'] = frameset.get_timestamp() / 1000
                if self.enable_color:
                    color_frame = frameset.get_color_frame()
                    data['color'] = np.asarray(color_frame.get_data())
                    t = color_frame.get_timestamp() / 1000
                    data['camera_capture_timestamp'] = t
                    # print('device', time.time() - t)
                    # print(color_frame.get_frame_timestamp_domain())
                if self.enable_depth:
                    depth_frame = frameset.get_depth_frame()
                    # proc_depth_frame = post_process_depth_frame(depth_frame, **self.depth_filter)
                    proc_depth_frame = self._process_depth(depth_frame)
                    data['depth'] = np.asarray(proc_depth_frame.get_data())
                    assert not np.all(data['depth'] == 0), 'depth all zero'
                    if self.is_ready:
                        data['intrinsics'] = self.get_intr_mat()
                        data['dist_coeffs'] = self.get_dist_coeff()
                        # if not cam_mat_saved and self.video_recorder.is_ready():
                        #     self.video_recorder.write_cam_mat(data['intrinsics'], data['dist_coeffs'])
                        #     cam_mat_saved = True
                    else:
                        warnings.warn('Intrinsics not ready, using identity matrix temporarily. Distance coefficients are temporarily set to zero.')
                        data['intrinsics'] = np.eye(3)
                        data['dist_coeffs'] = np.zeros(5)

                    data['extrinsics'] = self.extrinsics
                    

                if self.enable_infrared:
                    data['infrared'] = np.asarray(
                        frameset.get_infrared_frame().get_data())
                grab_time = time.time() - grab_start_time
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] Grab data time {grab_time}')

                # apply transform
                transform_start_time = time.time()

                put_data = data
                if self.transform is not None:
                    put_data = self.transform(dict(data))

                if self.put_downsample:                
                    # put frequency regulation
                    local_idxs, global_idxs, put_idx \
                        = get_accumulate_timestamp_idxs(
                            timestamps=[receive_time],
                            start_time=put_start_time,
                            dt=1/self.put_fps,
                            # this is non in first iteration
                            # and then replaced with a concrete number
                            next_global_idx=put_idx,
                            # continue to pump frames even if not started.
                            # start_time is simply used to align timestamps.
                            allow_negative=True
                        )

                    for step_idx in global_idxs:
                        put_data['step_idx'] = step_idx
                        # put_data['timestamp'] = put_start_time + step_idx / self.put_fps
                        put_data['timestamp'] = receive_time
                        # print(step_idx, data['timestamp'])
                        self.ring_buffer.put(put_data, wait=False)
                else:
                    step_idx = int((receive_time - put_start_time) * self.put_fps)
                    put_data['step_idx'] = step_idx
                    put_data['timestamp'] = receive_time
                    self.ring_buffer.put(put_data, wait=False)
                transform_time = time.time() - transform_start_time
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] Transform time {transform_time}')


                # signal ready
                if iter_idx == 0:
                    self.ready_event.set()
                
                # put to vis
                vis_data = data
                if self.vis_transform == self.transform:
                    vis_data = put_data
                elif self.vis_transform is not None:
                    vis_data = self.vis_transform(dict(data))
                self.vis_ring_buffer.put(vis_data, wait=False)
                
                # record frame
                rec_data = data
                rec_start_time = time.time()

                if self.recording_transform == self.transform:
                    rec_data = put_data
                elif self.recording_transform is not None:
                    rec_data = self.recording_transform(dict(data))

                if self.video_recorder.is_ready():
                    self.video_recorder.write_frame(rec_data['color'], 
                                                    data['depth'] if self.enable_depth else None, 
                                                    self.extrinsics,
                                                    frame_time=receive_time)
                rec_time = time.time() - rec_start_time
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] Record time {rec_time}')

                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] FPS {frequency}')

                # fetch command from queue
                cmd_start = time.time()
                try:
                    commands = self.command_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']
                    if cmd == Command.SET_COLOR_OPTION.value:
                        sensor = pipeline_profile.get_device().first_color_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                        # print('auto', sensor.get_option(rs.option.enable_auto_exposure))
                        # print('exposure', sensor.get_option(rs.option.exposure))
                        # print('gain', sensor.get_option(rs.option.gain))
                    elif cmd == Command.SET_DEPTH_OPTION.value:
                        sensor = pipeline_profile.get_device().first_depth_sensor()
                        option = rs.option(command['option_enum'])
                        value = float(command['option_value'])
                        sensor.set_option(option, value)
                    elif cmd == Command.START_RECORDING.value:
                        video_path = str(command['video_path'])
                        hdf5_path = str(command['hdf5_path'])
                        zarr_path = str(command['zarr_path'])
                        episode_id = command['episode_id']
                        cam_id = command['cam_id']
                        start_time = command['recording_start_time']
                        if start_time < 0:
                            start_time = None
                        self.video_recorder.start(video_path, hdf5_path, zarr_path, episode_id, cam_id, start_time=start_time)
                    elif cmd == Command.STOP_RECORDING.value:
                        self.video_recorder.stop()
                        # stop need to flush all in-flight frames to disk, which might take longer than dt.
                        # soft-reset put to drop frames to prevent ring buffer overflow.
                        put_idx = None
                    elif cmd == Command.RESTART_PUT.value:
                        put_idx = None
                        put_start_time = command['put_start_time']
                        # self.ring_buffer.clear()
                cmd_time = time.time() - cmd_start
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] Command time {cmd_time}')

                iter_idx += 1
                
                # perf
                t_end = time.time()
                duration = t_end - t_start
                frequency = np.round(1 / duration, 1)
                t_start = t_end
                if frequency < fps // 2:
                    print
                    warnings.warn(f'[{self.serial_number}] FPS {frequency} is much smaller than {fps}.')
                    print(f'debugging info of camera {self.serial_number}:')
                    print('wait_time:', wait_time)
                    print('grab_time:', grab_time)
                    print('transform_time:', transform_time)
                    # print('vis_time:', vis_time)
                    print('rec_time:', rec_time)
                    print('cmd_time:', cmd_time)
                if self.verbose:
                    print(f'[SingleRealsense {self.serial_number}] FPS {frequency}')

        finally:
            self.video_recorder.stop()
            rs_config.disable_all_streams()
            self.ready_event.set()
        
        if self.verbose:
            print(f'[SingleRealsense {self.serial_number}] Exiting worker process.')


def get_real_exporure_gain_white_balance():
    series_number = SingleRealsense.get_connected_devices_serial()
    with SharedMemoryManager() as shm_manager:
        with SingleRealsense(
            shm_manager=shm_manager,
            serial_number=series_number[4],
            enable_color=True,
            enable_depth=True,
            enable_infrared=False,
            put_fps=30,
            record_fps=30,
            verbose=True,
        ) as realsense:
            realsense.set_exposure(115, 64)
            realsense.set_white_balance(3100)
            realsense.set_contrast(60)

            for i in range(30):
                realsense.get()
                time.sleep(0.1)
            
            cv2.imshow('color', realsense.get()['color'])
            cv2.waitKey(0)

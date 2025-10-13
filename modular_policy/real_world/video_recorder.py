from typing import Optional, Callable, Generator
import numpy as np
import av
import os
import zarr
from modular_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs
import matplotlib.pyplot as plt
import h5py
import cProfile
import pstats
import io
from modular_policy.real_world.real_replay_buffer import ReplayBuffer
from numpy.lib.stride_tricks import as_strided

def read_video(
        video_path: str, 
        hdf5_path: str,
        dt: float,
        video_start_time: float=0.0, 
        start_time: float=0.0,
        img_transform: Optional[Callable[[np.ndarray], np.ndarray]]=None,
        thread_type: str="AUTO",
        thread_count: int=0,
        max_pad_frames: int=10
        ) -> Generator[np.ndarray, None, None]:
    """
    Reads a video file and yields frames at regular intervals determined by dt.

    Parameters:
    - video_path (str): Path to the video file.
    - dt (float): Interval between frames to extract, in seconds.
    - video_start_time (float, optional): Starting time offset for the video, defaults to 0.0.
    - start_time (float, optional): Start time for frame extraction, defaults to 0.0.
    - img_transform (Optional[Callable[[np.ndarray], np.ndarray]], optional): Function to apply to each frame.
    - thread_type (str, optional): Decoding thread type, defaults to "AUTO".
    - thread_count (int, optional): Number of threads for decoding, defaults to 0 (auto).
    - max_pad_frames (int, optional): Number of times to repeat the last frame, defaults to 10.

    Yields:
    - np.ndarray: The next frame in the video as a NumPy array.

    This function uses the PyAV library to open and decode the video. It iterates
    over each frame and checks if the frame's timestamp aligns with the specified interval (dt).
    If the timestamp aligns, the frame is transformed (if a transform is provided) and yielded.
    After all frames are processed, the last frame is repeated a specified number of times
    (max_pad_frames) to pad the video.
    """
    frame = None
    
    with av.open(video_path) as container, h5py.File(hdf5_path, 'r') as hdf5_file:
        
        stream = container.streams.video[0]
        stream.thread_type = thread_type
        stream.thread_count = thread_count
        
        next_global_idx = 0
        
        rgb_dataset = hdf5_file['rgb']
        depth_dataset = hdf5_file['depth']
        frame_time_dataset = hdf5_file['frame_time']
        
        for frame_idx, frame in enumerate(container.decode(stream)):
            # The presentation time in seconds for this frame.
            since_start = frame.time
            
            frame_time_hdf5 = frame_time_dataset[frame_idx]
            assert np.isclose(since_start, frame_time_hdf5)
            
            frame_time = video_start_time + since_start
            local_idxs, global_idxs, next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx
            )
            if len(global_idxs) > 0:
                array = frame.to_ndarray(format='rgb24')
                img = array
                    
                rgb_frame = rgb_dataset[frame_idx]
                depth_frame = depth_dataset[frame_idx]
            
                if img_transform is not None:
                    img = img_transform(array)
                for global_idx in global_idxs:
                    yield (img, rgb_frame, depth_frame)
                    
    # repeat last frame max_pad_frames times
    array = frame.to_ndarray(format='rgb24')
    img = array
    
    rgb_frame = rgb_dataset[frame_idx]
    depth_frame = depth_dataset[frame_idx]

    if img_transform is not None:
        img = img_transform(array)
    for i in range(max_pad_frames):
        yield (img, rgb_frame, depth_frame)


def read_cam_video(
        img_video_path: str, 
        depth_video_path: str,
        extrinsics_path: str,
        cam_intrinsics: np.ndarray,
        dt: float,
        video_start_time: float=0.0, 
        start_time: float=0.0,
        img_transform: Optional[Callable[[np.ndarray], np.ndarray]]=None,
        thread_type: str="AUTO",
        thread_count: int=0,
        fps: int=30,
        max_pad_frames: int=10,
        vectorized = False,
        debug=False
        ) -> Generator[np.ndarray, None, None]:
    """
    Reads a video file and yields frames at regular intervals determined by dt.

    Parameters:
    - video_path (str): Path to the video file.
    - dt (float): Interval between frames to extract, in seconds.
    - video_start_time (float, optional): Starting time offset for the video, defaults to 0.0.
    - start_time (float, optional): Start time for frame extraction, defaults to 0.0.
    - img_transform (Optional[Callable[[np.ndarray], np.ndarray]], optional): Function to apply to each frame.
    - thread_type (str, optional): Decoding thread type, defaults to "AUTO".
    - thread_count (int, optional): Number of threads for decoding, defaults to 0 (auto).
    - max_pad_frames (int, optional): Number of times to repeat the last frame, defaults to 10.

    Yields:
    - np.ndarray: The next frame in the video as a NumPy array.

    This function uses the PyAV library to open and decode the video. It iterates
    over each frame and checks if the frame's timestamp aligns with the specified interval (dt).
    If the timestamp aligns, the frame is transformed (if a transform is provided) and yielded.
    After all frames are processed, the last frame is repeated a specified number of times
    (max_pad_frames) to pad the video.
    """
    img_zarr = zarr.open(img_video_path, 'r')
    depth_zarr = zarr.open(depth_video_path, 'r')
    extrinsics_zarr = zarr.open(extrinsics_path, 'r')
    # boundaries = {'x_lower': -0.2, 'x_upper': 1, 'y_lower': -0.7, 'y_upper': 0.2, 'z_lower': -0.2, 'z_upper': 1}

    frame = None
    
    next_global_idx = 0
    video_dt = 1/fps
    rgb_dataset = img_zarr
    depth_dataset = depth_zarr
    # frame_time_dataset = hdf5_file['frame_time']
    since_start = 0

    if vectorized:
        total_frames = len(img_zarr)
        frame_times = np.arange(total_frames) * video_dt + video_start_time
        frame_indices = np.arange(total_frames)

        time_diffs = frame_times - start_time
        interval_indices = np.floor(time_diffs / dt).astype(np.int)

        # Now, find the change points in interval_indices to identify the first frame in each interval
        unique_intervals, unique_indices = np.unique(interval_indices, return_index=True)

        # unique_indices gives the index of the first occurrence of each unique value in interval_indices,
        # which corresponds to the first frame in each dt interval after start_time
        selected_indices = frame_indices[unique_indices]
        
        for frame_idx in selected_indices:
            img_frame = img_zarr[frame_idx]
            depth_frame = depth_zarr[frame_idx]
                        
            yield (_, img_frame, depth_frame)
        
        # Handle max_pad_frames
        if max_pad_frames > 0 and len(selected_indices) > 0:
            last_img_frame = img_zarr[selected_indices[-1]]
            last_depth_frame = depth_zarr[selected_indices[-1]]
                        
            for _ in range(max_pad_frames):
                yield (_, last_img_frame, last_depth_frame)

    else:
        for frame_idx, (img_frame, depth_frame, extrinsics_frame) in enumerate(zip(rgb_dataset, depth_dataset, extrinsics_zarr)):
            # The presentation time in seconds for this frame.
            # since_start = frame.time
            since_start = since_start + video_dt
            
            # frame_time_hdf5 = frame_time_dataset[frame_idx]
            # assert np.isclose(since_start, frame_time_hdf5)
            
            frame_time = video_start_time + since_start
            local_idxs, global_idxs, next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=start_time,
                dt=dt,
                next_global_idx=next_global_idx
            )
            if len(global_idxs) > 0:
                if abs(frame_idx - len(rgb_dataset)) < 100:
                    print(f'frame_idx: {frame_idx}, len(rgb_dataset): {len(rgb_dataset)}')
                array = img_frame
                img = img_frame
                    
                rgb_frame = rgb_dataset[frame_idx]
                depth_frame = depth_dataset[frame_idx]

                if img_transform is not None:
                    img = img_transform(array)
                for global_idx in global_idxs:
                    yield (img, rgb_frame, depth_frame)
                    
        # repeat last frame max_pad_frames times
        array = img_frame
        img = array

        if img_transform is not None:
            img = img_transform(array)
        for i in range(max_pad_frames):
            yield (img, img_frame, depth_frame)
            
            
class VideoRecorder:
    def __init__(self,
        fps,
        codec,
        input_pix_fmt,
        video_capture_resolution,
        video_zarr_path,
        # options for codec
        **kwargs
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        img_resolution = (video_capture_resolution[1], video_capture_resolution[0])
        self.img_shape = (*img_resolution, 3)
        self.depth_img_shape = img_resolution
        
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        
        self.record_mp4 = True
        self.record_hdf5 = False
        self.record_zarr = False
        
        # self.video_zarr_path = video_zarr_path
        
        # create replay buffer in read-write mode
        # replay_buffers = list()
        # if self.record_zarr:
        #     for zarr_path in video_zarr_path:
        #         replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a') 
        if video_zarr_path is not None:
            initial_max_frames=10000
            self.compressors = 'disk'
            attrs = {
                'fps':self.fps,
                'resolution': self.img_shape[:2],
                'num_cams':kwargs['num_cams']
            }
            self.replay_buffer = ReplayBuffer.create_from_path(video_zarr_path, mode='a', save_cam_data=True, attrs=attrs)   
            for i in range(kwargs['num_cams']):
                img_name = f'cam_{i}_img'
                depth_name = f'cam_{i}_depth'
                img_shape = (initial_max_frames,) + self.img_shape
                depth_shape = (initial_max_frames,) + self.depth_img_shape
                if img_name not in self.replay_buffer.data:
                    self.replay_buffer.data.zeros(name=img_name, shape=img_shape, dtype=np.uint8, 
                                        chunks=(1,) + self.img_shape,  # Adjust chunk size as needed
                                        compressors=self.compressors)
                if depth_name not in self.replay_buffer.data:
                    self.replay_buffer.data.zeros(name=depth_name, shape=depth_shape, dtype=np.float32,
                                            chunks=(1,) + self.depth_img_shape,  # Adjust chunk size as needed
                                            compressors=self.compressors)
                
        self.kwargs.pop('num_cams', None)
        self.hdf5_file = None
        
        # runtime set
        self._reset_state()
    
    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0
        # self.frame_time = 0
        self.frame_idx = 0
        
        self.episode = None
        self.episode_id = None
        self.cam_id = None
    
    @classmethod
    def create_h264(cls,
            fps,
            codec='h264',
            input_pix_fmt='rgb24',
            output_pix_fmt='yuv420p',
            crf=18,
            profile='high',
            video_capture_resolution=(1280, 720),
            video_zarr_path=None,
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                'crf': str(crf),
                'profile': profile
            },
            video_capture_resolution=video_capture_resolution,
            video_zarr_path=video_zarr_path,
            **kwargs
        )
        return obj


    def __del__(self):
        self.stop()

    def is_ready(self):
        if self.record_mp4:
            return self.stream is not None
        if self.record_hdf5:
            return self.hdf5_file is not None
        if self.record_zarr:
            return self.episode is not None
        
    def start(self, file_path, hdf5_path=None, zarr_path=None, episode_id=None, cam_id=None, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()
            
        self.start_time = start_time

        if self.record_mp4:
            self.container = av.open(file_path, mode='w')
            self.stream = self.container.add_stream(self.codec, rate=self.fps)
            codec_context = self.stream.codec_context
            for k, v in self.kwargs.items():
                setattr(codec_context, k, v)
        
        if self.record_hdf5:
            # check whether hdf5 file exists
            if os.path.exists(hdf5_path):
                os.remove(hdf5_path)
            self.hdf5_file = h5py.File(hdf5_path, 'a')
            self.hdf5_file.attrs['fps'] = self.fps
            self.hdf5_file.attrs['resolution'] = self.img_shape[:2]  # Store only width and height
            
            # Define chunk shape - choose a chunk shape that fits your data access pattern
            chunk_shape_rgb = (1, *self.img_shape)
            chunk_shape_depth = (1, *self.depth_img_shape)
            # chunk_shape_time = (1,)
            
            # Use gzip compression
            compression_opts = 4  # Adjust compression level here (1-9)
            
            # Preallocate a larger initial size for the HDF5 datasets
            initial_max_frames = 10000  # Adjust this based on expected maximum frames

            # self.rgb_dataset = self.hdf5_file.create_dataset(
            #     'rgb', shape=(0, *self.img_shape), maxshape=(None, *self.img_shape),
            #     chunks = True
            # )
            # self.depth_dataset = self.hdf5_file.create_dataset(
            #     'depth', shape=(0, *self.depth_img_shape), maxshape=(None, *self.depth_img_shape),
            #     chunks = True
            # )
            
            self.frame_idx = 0
            # self.frame_time_dataset = self.hdf5_file.create_dataset(
            #     'frame_time', shape=(0,), maxshape=(None,),
            #     chunks=chunk_shape_time, compression='gzip', compression_opts=compression_opts
            # )
        
            # self.rgb_dataset = self.hdf5_file.create_dataset('rgb', shape=(0, *self.img_shape), maxshape=(None, *self.img_shape))
            # self.depth_dataset = self.hdf5_file.create_dataset('depth', shape=(0, *self.depth_img_shape), maxshape=(None, *self.depth_img_shape))
            # self.frame_time_dataset = self.hdf5_file.create_dataset('frame_time', shape=(0,), maxshape=(None,))
            
            # self.frame_time = 0.0
        # if self.record_zarr:
        #     self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='a')
        if episode_id != None:
            self.episode = list()
            self.episode_id = episode_id
        if cam_id != None:
            self.cam_id = cam_id

    def write_frame(self, img: np.ndarray, depth_img: np.ndarray = None, extrinsics=None ,frame_time=None, profile=False):
        """
        Writes a given image as a frame into a video stream. If `start_time` is set, it calculates 
        the appropriate number of times the frame should be repeated based on the frame's timestamp 
        and the video's frames per second (fps) setting.

        Parameters:
        - img (np.ndarray): The image to be written as a frame. This should be a NumPy array.
        - frame_time (float, optional): The timestamp of the frame. If provided, it's used along with 
        `start_time` and `fps` to determine how many times the frame should be repeated. If `None`, 
        the frame is written once.

        Raises:
        - RuntimeError: If the method is called before the video stream is properly initialized 
        with `start()`.

        Details:
        - First, it checks if the stream is ready for writing frames. If not, it raises an error.
        - If `start_time` is not `None`, it calls `get_accumulate_timestamp_idxs` to calculate 
        `n_repeats` – the number of times the frame should be repeated based on its timestamp.
        - If the frame dimensions (`shape`) or data type (`dtype`) have not been set yet, it sets them 
        based on the first frame and also configures the stream dimensions.
        - Checks that the input image has the correct shape and data type as expected by the stream.
        - Converts the NumPy array `img` to an `av.VideoFrame`.
        - Encodes and writes the frame into the stream the number of times determined by `n_repeats`.
        """
        if profile:
            # Initialize profiler
            pr = cProfile.Profile()
            pr.enable()

        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
        
        n_repeats = 1
        if self.start_time is not None:
            # local_idxs is the indices of timestamps that fall into each dt interval
            local_idxs, global_idxs, self.next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1/self.fps,
                next_global_idx=self.next_global_idx
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)
        
        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h,w,c = img.shape
            if self.record_mp4:
                self.stream.width = w
                self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype
        
        if self.record_mp4:
            frame = av.VideoFrame.from_ndarray(
                img, format=self.input_pix_fmt)
            
        for i in range(n_repeats):
            if self.record_mp4:
                for packet in self.stream.encode(frame):
                    self.container.mux(packet)
                    
        if self.record_hdf5:
            try:
                self._write_to_hdf5(img, depth_img, n_repeats)
                    # Append new frames to the HDF5 datasets
                    # self.rgb_dataset.resize(self.rgb_dataset.shape[0] + 1, axis=0)
                    # self.rgb_dataset[-1] = img
                    # if depth_img is not None:
                    #     self.depth_dataset.resize(self.depth_dataset.shape[0] + 1, axis=0)
                    #     self.depth_dataset[-1] = depth_img     
                    # # Record the frame time
                    # self.frame_time_dataset.resize((self.frame_time_dataset.shape[0] + 1,))
                    # self.frame_time_dataset[-1] = self.frame_time
                    # self.frame_time += 1/self.fps
            except Exception as e:
                print("Error writing to HDF5 file:", e)
        
        if self.record_zarr:
            self._write_to_zarr(img, depth_img, extrinsics, n_repeats)

        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())

    def stop(self):
        if not self.is_ready():
            return
        
        profile = False
        if profile:
            # Initialize profiler
            pr = cProfile.Profile()
            pr.enable()
            
        if self.record_mp4:
            # Flush stream
            for packet in self.stream.encode():
                self.container.mux(packet)

            # Close the file
            self.container.close()
        
        if self.record_hdf5:
            data_dict = dict()
            for key in self.episode[0].keys():
                data_dict[key] = np.concatenate(
                    [x[key] for x in self.episode], axis=0)
            try:
                self.rgb_dataset = self.hdf5_file.create_dataset('img', data=data_dict['img'])
                self.depth_img_dataset = self.hdf5_file.create_dataset('depth', data=data_dict['depth'])
                self.hdf5_file.close()
            except Exception as e:
                print("Error closing HDF5 file:", e)
        
        if self.record_zarr:
            data_dict = dict()
            for key in self.episode[0].keys():
                data_dict[f'cam_{self.cam_id}_{key}'] = np.concatenate(
                    [x[key] for x in self.episode], axis=0)
            
            # pad the zeros to make the episode length same in data_dict
            # max_episode_length = max([x.shape[0] for x in data_dict.values()])
            # for key, value in data_dict.items():
            #     pad_length = max_episode_length - value.shape[0]
            #     if pad_length > 0:
            #         # value is of shape (n, h, w, c) or (n, h, w)
            #         pad_width = [(0, pad_length)] + [(0, 0)] * (len(value.shape) - 1)
            #         data_dict[key] = np.pad(value, pad_width=pad_width, mode='constant', constant_values=0)
            self.replay_buffer.add_cam_episode(data_dict, cam_id=self.cam_id, compressors=self.compressors)
            
        # reset runtime parameters
        self._reset_state()
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats()
            print(s.getvalue())



    def _write_to_hdf5(self, img, depth_img, n_repeats):
        # Efficient HDF5 writing
        # Calculate new size only once            
        # if n_repeats > 0:
        #     new_size = self.rgb_dataset.shape[0] + n_repeats
        #     self.rgb_dataset.resize(new_size, axis=0)
        #     if depth_img is not None:
        #         self.depth_dataset.resize(new_size, axis=0)

        #     final_frame_idx = self.frame_idx + n_repeats
        #     repeated_rgb_images = np.tile(img, (n_repeats, 1, 1, 1))  # Adjust the shape as necessary
        #     self.rgb_dataset[self.frame_idx:final_frame_idx] = repeated_rgb_images

        #     if depth_img is not None:
        #         repeated_depth_images = np.tile(depth_img, (n_repeats, 1, 1, 1))  # Adjust the shape as necessary
        #         self.depth_dataset[self.frame_idx:final_frame_idx] = repeated_depth_images
                
        #     self.frame_idx = final_frame_idx
        data = {
            'img': repeat_image_efficiently(img, n_repeats),
            'depth': repeat_image_efficiently(depth_img, n_repeats),
        }
        self.episode.append(data)

    def _write_to_zarr(self, img, depth_img, extrinsics, n_repeats):
        data = {
            'img': repeat_image_efficiently(img, n_repeats),
            'depth': repeat_image_efficiently(depth_img, n_repeats),
            'extrinsics': repeat_image_efficiently(extrinsics, n_repeats) if extrinsics is not None else None,
        }
        self.episode.append(data)
        
    def write_cam_mat(self, intrinsics, dist_coeffs):
        self.replay_buffer.root.attrs[f'cam_{self.cam_id}_intrinsics'] = intrinsics.tolist()
        self.replay_buffer.root.attrs[f'cam_{self.cam_id}_dist_coeffs'] = dist_coeffs.tolist()
   
def repeat_image_efficiently(image, n):
    # Create a new shape and new strides
    new_shape = (n,) + image.shape
    new_strides = (0,) + image.strides
    
    # Use as_strided to create repeated views of the same data
    repeated_images_view = as_strided(image, shape=new_shape, strides=new_strides)

    # Make a copy to ensure separate memory allocation
    repeated_images_copy = np.copy(repeated_images_view)
    
    return repeated_images_copy
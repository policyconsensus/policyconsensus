from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property
import dask.array as da
import uuid

def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    
    random_id = uuid.uuid4()
    tmp_key = f"tmp_key_{random_id}"

    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


def extract_data_for_episodes(data, meta, num_episodes):
    """Extracts the data for a given number of episodes."""

    # Ensure we don't ask for more episodes than available
    num_episodes = min(num_episodes, len(meta['episode_ends']))
    print('num_episodes', num_episodes)

    # Calculate the ending index for the last episode we want to extract
    last_episode_end = meta['episode_ends'][num_episodes - 1]


    extracted_data = {}
    for key, value in data.items():
        extracted_data[key] = value[0:last_episode_end]

    # Update meta's episode_ends to only contain ends for the number of episodes
    extracted_meta = {
        'episode_ends': meta['episode_ends'][:num_episodes]
    }

    return {'data': extracted_data, 'meta': extracted_meta}


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """
    def __init__(self, 
            root: Union[zarr.Group, 
            Dict[str,dict]],
            save_cam_data=False,
            attrs=None):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        
        def check_episode_length(value, episode_length, parent_key='', root=None):

            if isinstance(value, zarr.Group):
                for k, v in value.items():
                    new_key = f"{parent_key}/{k}" if parent_key else k  # Properly concatenate the parent key
                    check_episode_length(v, episode_length, parent_key=new_key, root=root)
            else:
                # Only print and prompt if lengths do not match
                if value.shape[0] != episode_length:
                    print(f"Inconsistent episode length for key: {parent_key} (expected {episode_length}, found {value.shape[0]})")
                    user_input = input(f"Do you want to (1) delete {parent_key} in the dataset, (2) truncate it to the correct length, or (3) leave it unchanged? (1/2/3): ").strip().lower()

                    if user_input == '1':
                        print(f"Deleting dataset '{parent_key}'...")
                        del root['data'][parent_key]
                    elif user_input == '2':
                        print(f"Truncating dataset '{parent_key}' to length {episode_length}...")
                        root['data'][parent_key].resize((episode_length,) + value.shape[1:])  # Resizing to the correct length
                        print(f"Dataset '{parent_key}' truncated to {episode_length}.")
                    else:
                        print(f"Invalid input. Dataset '{parent_key}' not modified.")
                        raise ValueError(f"Dataset '{parent_key}' has an inconsistent episode length and was not modified.")

                # assert value.shape[0] == episode_length, f"Inconsistent episode length for key: {key} (expected {episode_length}, found {value.shape[0]})"

        if not save_cam_data and len(root['meta']['episode_ends']) > 0:
            
            episode_length = root['meta']['episode_ends'][-1]
            print(f"Expected episode length: {episode_length}")
            print("Checking lengths of datasets...")

            for key, value in root['data'].items():
                # check_episode_length(value, episode_length)
                check_episode_length(value, episode_length, parent_key=key, root=root)

        self.root = root
    
    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None, **kwargs):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        if 'episode_ends' not in meta:
            episode_ends = meta.zeros('episode_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        if kwargs.get('save_cam_data'):
            attrs = kwargs.get('attrs')
            root.attrs['fps'] = attrs['fps']
            root.attrs['resolution'] = attrs['resolution']
            root.attrs['num_cams'] = attrs['num_cams'] 

            # Retrieve the number of cameras from the attributes
            num_cams = attrs['num_cams']

            # Pre-filter keys for each camera to avoid nested loop checks
            cam_data_keys = {cam_id: [key for key in root['data'].keys() if f"cam_{cam_id}" in key] for cam_id in range(num_cams)}
            # Validate each camera's data
            for cam_id, keys in cam_data_keys.items():
                episode_ends_key = f'cam_{cam_id}_episode_ends'
                if episode_ends_key not in root['meta']:
                    cam_episode_ends = root['meta'].zeros(episode_ends_key, shape=(0,), dtype=np.int64,
                        compressor=None, overwrite=False)
                # else:
                #     # Retrieve the episode ends for the current camera
                #     cam_episode_ends = root['meta'][f'cam_{cam_id}_episode_ends']

                # for key in keys:
                #     # Check the shape of the data array for the current key
                #     assert root['data'][key].shape[0] == cam_episode_ends[-1], f"Mismatch in episode length for cam_{cam_id}"

        return cls(root=root)
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64)
            }
        }
        return cls(root=root)
    
    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        root = None
        num_episodes = int(kwargs['num_episodes']) if kwargs.get('num_episodes') is not None else None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root['meta'].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                if ',' in key and 'action' in key:
                    key_list = [k.strip() for k in key.split(',')]
                    arr = np.hstack([src_root['data'][k][:] for k in key_list])
                    data['action'] = arr[:]
                    continue
                arr = src_root['data'][key]
                data[key] = arr[:]
            if num_episodes is not None:
                root = extract_data_for_episodes(data, meta, num_episodes)
            else:
                root = {
                    'meta': meta,
                    'data': data
                }
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                keys = src_root['data'].keys()
            for key in keys:
                value = src_root['data'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
        buffer = cls(root=root)
        return buffer

    @classmethod
    def read_from_store(cls, src_store, keys=None, 
                chunks: Optional[Dict[str, tuple]] = dict(),
                compressors: Union[dict, str, numcodecs.abc.Codec] = dict(),
                if_exists='replace', **kwargs):
        """
        Reads data from the source store into the replay buffer without copying the data.
        
        This method reads from a Zarr store, linking the datasets without copying them into memory. 
        The resulting buffer directly uses the datasets from the source store.
        
        Parameters:
        - src_store: Source Zarr store to read data from.
        - store: Destination Zarr store (optional). If not provided, it will link data from the source store.
        - keys: Optional keys to restrict the datasets being read.
        - chunks: Optional chunk configuration.
        - compressors: Optional compressor settings.
        - if_exists: Determines the behavior if datasets already exist ('replace', 'skip', etc.).
        
        Returns:
        - ReplayBuffer instance populated with the data from the src_store without copying it.
        """
        
        src_root = zarr.group(src_store)
        
        # Return a ReplayBuffer instance
        return cls(root=src_root)
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), 'r')
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)
    

    # ============= save methods ===============
    def save_to_store(self, store, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            if_exists='replace', 
            **kwargs):
        
        root = zarr.group(store)
        if self.backend == 'zarr':
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
        else:
            meta_group = root.create_group('meta', overwrite=True)
            # save meta, no chunking
            for key, value in self.root['meta'].items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape)
        
        # save data, chunk
        data_group = root.create_group('data', overwrite=True)
        for key, value in self.root['data'].items():
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = data_group.array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        return store

    def save_to_path(self, zarr_path,             
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(), 
            if_exists='replace', 
            **kwargs):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, 
            compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor='default'):
        if compressor == 'default':
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, 
                shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == 'disk':
            compressor = numcodecs.Blosc('zstd', clevel=5, 
                shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, 
            compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = 'nil'
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == 'nil':
            cpr = cls.resolve_compressor('default')
        return cpr
    
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks
    
    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == 'zarr':
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape,
                    overwrite=True)
        else:
            meta_group.update(np_data)
        
        return meta_group
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']

    @property
    def cam_episode_ends(self):

        # cam_episode_ends = []
        # for i in range(len(self.meta)):
        #     if f'cam_{i}_episode_ends' in self.meta:
        #         cam_episode_ends.append(self.meta[f'cam_{i}_episode_ends'])

        # if len(cam_episode_ends) == 0:
        #     return [self.meta['episode_ends']]
        # else:
        #     return cam_episode_ends
        cam_episode_ends_list = [self.meta[f'cam_{i}_episode_ends'] for i in range(len(self.meta)) if f'cam_{i}_episode_ends' in self.meta]
        return cam_episode_ends_list if cam_episode_ends_list else [self.meta['episode_ends']]

    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(self.episode_ends)
        
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]

    @property
    def n_episodes(self):
        return len(self.episode_ends)
    

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):
        assert(len(data) > 0)
        is_zarr = (self.backend == 'zarr')
        curr_len = self.n_steps
        
        def get_episode_length(value):
            if isinstance(value, dict):
                lengths = [get_episode_length(v) for v in value.values()]
                if len(set(lengths)) > 1:
                    raise ValueError("Inconsistent episode lengths in the nested dictionary.")
                return lengths[0]
            else:
                assert(len(value.shape) >= 1)
                return len(value)

        episode_length = get_episode_length(data)
        new_len = curr_len + episode_length

        def add_data(key, value, parent_arr):
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    add_data(f"{key}/{sub_key}", sub_value, parent_arr)
            else:
                new_shape = (new_len,) + value.shape[1:]
                if key not in parent_arr:
                    if is_zarr:
                        cks = self._resolve_array_chunks(chunks=chunks, key=key, array=value)
                        cpr = self._resolve_array_compressor(compressors=compressors, key=key, array=value)
                        arr = parent_arr.zeros(name=key, shape=new_shape, chunks=cks, dtype=value.dtype, compressor=cpr)
                    else:
                        arr = np.zeros(shape=new_shape, dtype=value.dtype)
                        parent_arr[key] = arr
                else:
                    arr = parent_arr[key]
                    assert(value.shape[1:] == arr.shape[1:])
                    if is_zarr:
                        arr.resize(new_shape)
                    else:
                        arr.resize(new_shape, refcheck=False)
                arr[-value.shape[0]:] = value
                
        for key, value in data.items():
            add_data(key, value, self.data)
        
        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(self.meta, 'episode_ends', 
                    chunk_length=int(episode_ends.shape[0] * 1.5))

            
    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert(len(episode_ends) > 0)
        
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]

        def resize_array(value, new_shape):
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)

        def drop_data(data):
            for key, value in data.items():
                if isinstance(value, zarr.Group):
                    drop_data(value)
                else:
                    new_shape = (start_idx,) + value.shape[1:]
                    resize_array(value, new_shape)

        drop_data(self.data)

        if is_zarr:
            self.episode_ends.resize(len(episode_ends) - 1)
        else:
            self.episode_ends.resize(len(episode_ends) - 1, refcheck=False)

    def drop_cam_episode(self):
        is_zarr = (self.backend == 'zarr')
        cam_episode_ends = self.cam_episode_ends[:].copy()
        assert(len(cam_episode_ends) > 0)
        start_idx = 0
        for cam_ep_ends in cam_episode_ends:
            if len(cam_ep_ends) > 1:
                start_idx = cam_ep_ends[-2]

                for key, value in self.data.items():
                    new_shape = (start_idx,) + value.shape[1:]
                    if is_zarr:
                        value.resize(new_shape)
                    else:
                        value.resize(new_shape, refcheck=False)

        for cam_id in range(len(cam_episode_ends)):          
            if len(cam_episode_ends[cam_id]) > 0:
                if is_zarr:
                    self.cam_episode_ends[cam_id].resize(len(cam_episode_ends[cam_id])-1)
                else:
                    self.cam_episode_ends[cam_id].resize(len(cam_episode_ends[cam_id])-1, refcheck=False)

    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result
    
    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result


    def add_cam_episode(self, 
            data: Dict[str, np.ndarray], 
            cam_id: int, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):
        """
        Adds an episode for a specific camera to the replay buffer.
        """

        assert(len(data) > 0)
        initial_max_frames = 10000

        is_zarr = (self.backend == 'zarr')
        
        episode_ends_key = f'cam_{cam_id}_episode_ends'
        cam_episode_ends = self.meta[episode_ends_key]
        
        curr_len = 0 if len(cam_episode_ends) == 0 else cam_episode_ends[-1]

        episode_length = None
        for key, value in data.items():
            assert(len(value.shape) >= 1)
            if episode_length is None:
                episode_length = len(value)
            else:
                assert(episode_length == len(value))
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # new_shape = (initial_max_frames,) + value.shape[1:]
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, 
                        shape=new_shape, 
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr)                    
                else:
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr

            else:
                arr = self.data[key]
                assert(value.shape[1:] == arr.shape[1:])
                if arr.shape[0] < new_len:
                    arr.resize(new_shape)
            
            arr[curr_len:new_len] = value  # Write data directly without concatenation
            
        
        # append to episode ends
        if is_zarr:
            cam_episode_ends.resize(cam_episode_ends.shape[0] + 1)
        else:
            cam_episode_ends.resize(cam_episode_ends.shape[0] + 1, refcheck=False)
        cam_episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if cam_episode_ends.chunks[0] < cam_episode_ends.shape[0]:
                rechunk_recompress_array(self.meta, episode_ends_key, 
                    chunk_length=int(cam_episode_ends.shape[0] * 1.5))

    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == 'zarr'
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks
    
    def set_chunks(self, chunks: dict):
        assert self.backend == 'zarr'
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == 'zarr'
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == 'zarr'
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)


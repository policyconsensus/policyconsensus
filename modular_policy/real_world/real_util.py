import numpy as np
import math
import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
import cProfile
import open3d as o3d
from typing import Tuple, Dict, Union, List


import io
import pstats
import dask.array as da
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ExifTags
import re
from scipy.spatial.transform import Rotation as R, Slerp
import scipy.interpolate as si
from klampt import WorldModel
from klampt.io import numpy_convert, open3d_convert

from modular_policy.model.common.rotation_transformer import RotationTransformer

from typing import Tuple, Dict, Union

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False


ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ImgNorm_inv = tvf.Compose([tvf.Normalize((-0.5/0.5, -0.5/0.5, -0.5/0.5), (1/0.5, 1/0.5, 1/0.5))])
TorchNorm = tvf.Compose([tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def get_image_transform(
        input_res: Tuple[int,int]=(1280,720), 
        output_res: Tuple[int,int]=(640,480), 
        bgr_to_rgb: bool=False):

    iw, ih = input_res
    ow, oh = output_res
    rw, rh = None, None
    interp_method = cv2.INTER_AREA

    if (iw/ih) >= (ow/oh):
        # input is wider
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    
    w_slice_start = (rw - ow) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = (rh - oh) // 2
    h_slice = slice(h_slice_start, h_slice_start + oh)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # resize
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        # crop
        img = img[h_slice, w_slice, c_slice]
        return img
    return transform


def real_obs_to_policy_obs(
    real_obs: Dict[str, np.ndarray],
    vis_size: Tuple[int, int] = (96, 128),
) -> Dict[str, np.ndarray]:
    
    axis_angle_to_6drot = RotationTransformer(
        from_rep='axis_angle', to_rep='rotation_6d')
    rgb_transform = tvf.Resize(vis_size, 
        interpolation=tvf.InterpolationMode.BILINEAR,
        antialias=True)
    depth_transform = tvf.Resize(vis_size, 
        interpolation=tvf.InterpolationMode.NEAREST,
        antialias=True)

    policy_obs = dict()
    for key, value in real_obs.items():
        
        # handle rgb and depth
        if key.endswith('_color'):
            if value.ndim == 4 and value.shape[-1] == 3:
                new_rgb = value.transpose(0, 3, 1, 2)           # (N, H, W, 3) -> (N, 3, H, W)
            elif value.ndim == 3 and value.shape[-1] == 3:
                new_rgb = value.transpose(2, 0, 1)              # (H, W, 3) -> (3, H, W)
            new_rgb = torch.from_numpy(new_rgb)
            new_rgb = rgb_transform(new_rgb)
            new_rgb = new_rgb.cpu().numpy()
            policy_obs[key] = new_rgb

        elif key.endswith('_depth'):
            new_depth = torch.from_numpy(value / 1000.0)        # mm -> m
            new_depth = depth_transform(new_depth)
            if value.ndim == 3:
                new_depth = np.expand_dims(new_depth, axis=1)   # (N, H, W) -> (N, 1, H, W)
            elif value.ndim == 2:
                new_depth = np.expand_dims(new_depth, axis=0)   # (H, W) -> (1, H, W)
            policy_obs[key] = new_depth

        # 6drot is better for policy
        elif key == 'robot_eef_pose':
            trans = value[..., 0:3]
            rot = value[..., 3:6]
            gripper = value[..., 6:]
            new_rot = axis_angle_to_6drot.forward(rot)
            new_eef = np.concatenate([
                trans, new_rot, gripper
            ], axis=-1)
            policy_obs[key] = value
            policy_obs['robot_eef_9dpose'] = new_eef

        # others
        else:
            policy_obs[key] = value

    return policy_obs
    

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        task_conf=None,
        kin_helper=None,
        real_obs_info={},
        ) -> Dict[str, np.ndarray]:
    raise DeprecationWarning
    rotation_transformer = RotationTransformer(
        from_rep='axis_angle', to_rep='rotation_6d')
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    # if 'robot_eef_pose' in env_obs.keys() and 'keypoint' in env_obs.keys():
    #     obs_dict_np = {'obs': env_obs['keypoint_in_world']}
    #     return obs_dict_np
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'robot_state')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        if type == 'depth':
            this_imgs_in = env_obs[key]
            t,hi,wi = this_imgs_in.shape
            ho,wo = shape
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                out_imgs = np.stack([cv2.resize(x, (wo,ho), cv2.INTER_LANCZOS4) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint16:
                    out_imgs = out_imgs.astype(np.float32) / 1000
            obs_dict_np[key] = out_imgs
        elif type == 'robot_state':

            this_data_in = env_obs[key]
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
            if key =='robot_eef_pose':
                this_data_in = _convert_actions(this_data_in, rotation_transformer, 'cartesian_action')
                this_data_in = this_data_in[:, :shape_meta['obs']['robot_eef_pose'].shape[0]]
            obs_dict_np[key] = this_data_in

        elif type == 'feature':
            mast3r_size = max(shape_meta['obs']['camera_features']['original_shape'])
            test_feature_view_key = real_obs_info['feature_view_key']

            view_image = env_obs[f'{test_feature_view_key}_color'] # (T, H ,W, C)
            color_seq = view_image


            mast3r_imgs = load_images_mast3r(color_seq, size=mast3r_size, verbose=False)
            c,h,w = tuple(shape_meta['obs']['camera_features']['shape'])

            obs_dict_np[key] = F.interpolate(torch.concat([ImgNorm_inv(mast3r_imgs[i]['img']) for i in range(len(mast3r_imgs))]),size=(h,w), mode='area').cpu().numpy()

    return obs_dict_np



def _convert_actions(raw_actions, rotation_transformer, action_key):        
    act_num, act_dim = raw_actions.shape
    if act_dim == 2:
        return raw_actions
    is_bimanual = (act_dim == 14) or (act_dim == 12) 
    if is_bimanual:
        raw_actions = raw_actions.reshape(act_num * 2, act_dim // 2)
    
    if action_key == 'cartesian_action' or action_key == 'robot_eef_pose':
        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    elif action_key == 'joint_action':
        pass
    else:
        raise RuntimeError('unsupported action_key')
    if is_bimanual:
        proc_act_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(act_num, proc_act_dim * 2)
    actions = raw_actions
    return actions


def load_images_mast3r(folder_or_list, size, square_ok=False, verbose=True, device='cuda'):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list
    elif isinstance(folder_or_list, np.ndarray):
        # Check if the NumPy array is a batch of images (N, W, H, C)
        if len(folder_or_list.shape) == 4:  # Batch of images
            N, W, H, C = folder_or_list.shape
            if verbose:
                print(f'>> Loading a batch of {N} images from a numpy array')
            folder_content = folder_or_list
            root = ''
        else:
            raise ValueError(f'Invalid numpy array shape: {folder_or_list.shape}. Expected (N, W, H, C).')
    elif isinstance(folder_or_list, da.Array):
        # Handle Dask array (batch of images)
        if len(folder_or_list.shape) == 4:  # Batch of images
            N, W, H, C = folder_or_list.shape
            if verbose:
                print(f'>> Loading a batch of {N} images from a Dask array')
            folder_content = folder_or_list
            root = ''
        else:
            raise ValueError(f'Invalid Dask array shape: {folder_or_list.shape}. Expected (N, W, H, C).')
    elif isinstance(folder_or_list, torch.Tensor):
        # If it's a PyTorch tensor, skip resizing and cropping
        if len(folder_or_list.shape) == 4:
            N, C, H, W = folder_or_list.shape
            if verbose:
                print(f'>> Loading a batch of {N} images from a torch Tensor')
            folder_content = folder_or_list
            root = ''
        else:
            raise ValueError(f'Invalid tensor shape: {folder_or_list.shape}. Expected (N, C, H, W).')
    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path_or_img in folder_content:
        if isinstance(path_or_img, str) and not path_or_img.lower().endswith(supported_images_extensions):
            # Skip non-image files if input is a folder
            continue
        
        # Load the image from file or use the preloaded NumPy array
        if isinstance(path_or_img, str):
            img = exif_transpose(PIL.Image.open(os.path.join(root, path_or_img))).convert('RGB')
        elif isinstance(path_or_img, np.ndarray):
            # Convert the NumPy array (assuming it has shape HxWx3) to a PIL image
            img = PIL.Image.fromarray(path_or_img.astype(np.uint8))
        elif isinstance(folder_or_list, da.Array):
            # Convert the Dask array slice to a NumPy array (using .compute()) and then to a PIL image
            img = PIL.Image.fromarray(path_or_img.compute().astype(np.uint8))
        elif isinstance(path_or_img, torch.Tensor):
            img = path_or_img
        else:
            raise ValueError(f'Unsupported image type: {type(path_or_img)}')
        
        if isinstance(path_or_img, torch.Tensor):
            imgs.append(dict(img=TorchNorm(img)[None].to(device), true_shape=np.int32(
                [img.shape[-2:]]), idx=len(imgs), instance=str(len(imgs))))

        else:
            W1, H1 = img.size
            if size == 224:
                # resize short side to 224 (then crop)
                img = _resize_pil_image(img, round(size * max(W1/H1, H1/W1)))
            else:
                # resize long side to 512
                img = _resize_pil_image(img, size)
            W, H = img.size
            cx, cy = W//2, H//2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
                if not (square_ok) and W == H:
                    halfh = 3*halfw/4
                img = img.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))

            W2, H2 = img.size
            if verbose:
                print(f' - adding image with resolution {W1}x{H1} --> {W2}x{H2}')
                
            imgs.append(dict(img=ImgNorm(img)[None].to(device), true_shape=np.int32(
                [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, f'No images found in {root}'
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs


def interpolate_poses(start_pose, final_pose, speed, translation_tolerance=0.02, rotation_tolerance=np.deg2rad(5)):
    # Split the start and final poses into translation and rotation components
    start_translation, start_rotation = start_pose[:3], start_pose[3:]
    final_translation, final_rotation = final_pose[:3], final_pose[3:]
    
    # Calculate the total distance for translation
    translation_distance = np.linalg.norm(final_translation - start_translation)
    
    # Calculate rotation angle difference using quaternion distance
    start_quat = R.from_rotvec(start_rotation).as_quat()
    final_quat = R.from_rotvec(final_rotation).as_quat()
    rotation_angle_difference = R.from_quat([start_quat]).inv() * R.from_quat([final_quat])
    rotation_angle_difference = rotation_angle_difference.magnitude()
    
    # Check if start and final poses are close
    if translation_distance <= translation_tolerance and rotation_angle_difference <= rotation_tolerance:
        # Poses are considered close, return an indication and skip interpolation
        return [], 0, True
    
    # Calculate the number of steps required based on the given speed
    total_steps = max(int(np.ceil(translation_distance / speed)), 1)
    
    # Setup for Slerp and linear interpolation
    times = [0, 1]  # Start and end times
    key_rotations = R.from_quat([start_quat, final_quat])
    slerp = Slerp(times, key_rotations)
    proportional_times = np.linspace(0, 1, total_steps + 2)[1:-1]
    interp_func = si.interp1d([0, 1], np.vstack([start_translation, final_translation]), axis=0)
    
    # Perform interpolations
    interpolated_rotations = slerp(proportional_times)
    interpolated_translations = interp_func(proportional_times)
    
    # Combine interpolated translations and rotations
    interpolated_poses = []
    for i in range(total_steps):
        pose = np.hstack([interpolated_translations[i], interpolated_rotations[i].as_rotvec()])
        interpolated_poses.append(pose)
    
    return interpolated_poses, total_steps, False


def policy_action_to_env_action(raw_actions, cur_eef_pose_6d=None, init_tool_pose_9d=None, 
                                tool_in_eef=None, action_mode='eef', debug=False):
    '''
    Convert policy actions to environment actions based on the given action mode.
    - cur_eef_pose_6d: Current end-effector pose in 6D (position + axis-angle).
    - init_tool_pose_9d: Initial tool pose in 9D (position + 6D rotation).
    - action_mode: 'eef' for end-effector control, 'joint' for joint control.
    - tool_in_eef: Transformation of the tool in the end-effector frame.
    - robot_base_pose_in_world: Transformation of the robot base in the world frame.
    - cam_extrinsic: Camera extrinsics in the world frame.
    - num_bots: Number of robots (defaults to 1).
    - debug: Whether to enable debug logging and assertions.
    '''
    if action_mode == 'eef' and raw_actions.shape[1] == 3:
        assert cur_eef_pose_6d is not None
        # raw_actions is of shape (T, 3). cur_eef_pose is of shape (6)
        # convert cur_eef_pose to (T, 6) and then concatenate with raw_actions
        raw_actions = np.concatenate([raw_actions, cur_eef_pose_6d[None,3:].repeat(raw_actions.shape[0], axis=0)], axis=1)
        return raw_actions
    elif action_mode == 'eef' and raw_actions.shape[1] == 7:
        return raw_actions
    elif action_mode == 'eef':
        if init_tool_pose_9d is not None:            
            pred_eef_pose = []
            
            tool_in_base = _9d_to_homo_action(raw_actions)
            pred_eef_pose = tool_in_base@np.linalg.inv(tool_in_eef) 
                
            pred_eef_pose_6d = _homo_to_6d_axis_angle(pred_eef_pose)

            if debug:
                tool_in_base_ls = np.array(tool_in_base_ls)
            return pred_eef_pose_6d 
        else:
            return eef_action_to_6d(raw_actions)
        
    elif action_mode == 'joint':
        return raw_actions


def _9d_to_homo_action(raw_actions):
    
    trans = raw_actions[:, :3]  
    rot_vec = raw_actions[:, 3:]  
    rot_mat = convert_rotation_6d_to_matrix(rot_vec)
    
    batch_size = raw_actions.shape[0]
    homo_action = np.zeros((batch_size, 4, 4))  # Initialize homogeneous matrix

    homo_action[:, :3, :3] = rot_mat  # Set rotation part
    homo_action[:, :3, 3] = trans  # Set translation part
    homo_action[:, 3, 3] = 1.0  # Set bottom-right part to 1 (for homogeneous coordinates)

    return homo_action


def _homo_to_6d_axis_angle(homo_action):
    trans = homo_action[:, :3, 3]  
    rot_mat = homo_action[:, :3, :3]  
    axis_angle = convert_matrix_to_axis_angle(rot_mat)
    
    batch_size = homo_action.shape[0]
    raw_actions = np.zeros((batch_size, 6))  # Initialize raw action

    raw_actions[:, :3] = trans  # Set translation part
    raw_actions[:, 3:] = axis_angle  # Set rotation part

    return raw_actions


def eef_action_to_6d(raw_actions):
    rotation_transformer = RotationTransformer(
        from_rep='rotation_6d', to_rep='axis_angle')
    act_num, act_dim = raw_actions.shape
    is_bimanual = (act_dim == 18) or (act_dim == 20)
    if is_bimanual:
        raw_actions = raw_actions.reshape(act_num * 2, act_dim // 2)
    
    pos = raw_actions[...,:3]
    rot = raw_actions[...,3:9]
    gripper = raw_actions[...,9:]
    rot = rotation_transformer.forward(rot)
    raw_actions = np.concatenate([
        pos, rot, gripper
    ], axis=-1).astype(np.float32)
    if is_bimanual:
        proc_act_dim = raw_actions.shape[-1]
        raw_actions = raw_actions.reshape(act_num, proc_act_dim * 2)
    actions = raw_actions
    return actions


def convert_matrix_to_axis_angle(matrix):
    rotation_transformer = RotationTransformer(
        from_rep='matrix', to_rep='axis_angle')
    axis_angle = rotation_transformer.forward(matrix)
    return axis_angle


def convert_rotation_6d_to_matrix(rot_6d):
    rotation_transformer = RotationTransformer(
        from_rep='rotation_6d', to_rep='matrix')
    rot_mat = rotation_transformer.forward(rot_6d)
    return rot_mat


_transform_cache = {}
_rotation_transformer_cache = {}

def _get_cached_transform(vis_size: Tuple[int, int], transform_type: str):
    """Get cached transform objects to avoid repeated creation."""
    cache_key = (vis_size, transform_type)
    if cache_key not in _transform_cache:
        if transform_type == 'rgb':
            _transform_cache[cache_key] = tvf.Resize(vis_size, 
                interpolation=tvf.InterpolationMode.BILINEAR,
                antialias=True)
        elif transform_type == 'depth':
            _transform_cache[cache_key] = tvf.Resize(vis_size, 
                interpolation=tvf.InterpolationMode.NEAREST,
                antialias=True)
    return _transform_cache[cache_key]

def _get_cached_rotation_transformer(from_rep: str, to_rep: str):
    """Get cached rotation transformer to avoid repeated creation."""
    cache_key = (from_rep, to_rep)
    if cache_key not in _rotation_transformer_cache:
        _rotation_transformer_cache[cache_key] = RotationTransformer(
            from_rep=from_rep, to_rep=to_rep)
    return _rotation_transformer_cache[cache_key]

def _fast_resize_opencv(images: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Fast image resizing using OpenCV - 3-5x faster than torchvision"""
    h, w = target_size
    # Use INTER_AREA for downsampling (better quality), INTER_LINEAR for upsampling
    interpolation = cv2.INTER_AREA  # Best for downsampling (480x640 -> 96x128)
    
    if images.ndim == 4:  # (T, H, W, C)
        return np.array([cv2.resize(img, (w, h), interpolation=interpolation) 
                        for img in images], dtype=np.float32)
    else:  # (H, W, C)
        return cv2.resize(images, (w, h), interpolation=interpolation).astype(np.float32)

def real_obs_to_policy_obs_batched(
    real_obs_list: List[Dict[str, np.ndarray]],
    vis_size: Tuple[int, int] = (96, 128),
    use_fast_resize: bool = True,
) -> List[Dict[str, np.ndarray]]:
    """
    Batched version of real_obs_to_policy_obs for processing multiple episodes efficiently.
    
    Args:
        real_obs_list: List of observation dictionaries
        vis_size: Target size for image resizing
        
    Returns:
        List of processed observation dictionaries
    """
    if not real_obs_list:
        return []
        
    # Use cached transforms
    axis_angle_to_6drot = _get_cached_rotation_transformer('axis_angle', 'rotation_6d')
    rgb_transform = _get_cached_transform(vis_size, 'rgb')
    depth_transform = _get_cached_transform(vis_size, 'depth')
    
    # Get all keys from first observation (assume all episodes have same keys)
    all_keys = set(real_obs_list[0].keys())
    # Separate keys by type for batch processing
    rgb_keys = [k for k in all_keys if k.endswith('_color')]
    depth_keys = [k for k in all_keys if k.endswith('_depth')]
    
    policy_obs_list = []
    
    # Process RGB images in batches if multiple episodes have same keys
    rgb_batches = {}
    depth_batches = {}
    
    # Collect all RGB and depth data for batch processing
    for episode_idx, real_obs in enumerate(real_obs_list):
        for key in rgb_keys:
            if key in real_obs:
                if key not in rgb_batches:
                    rgb_batches[key] = []
                rgb_batches[key].append((episode_idx, real_obs[key]))
                
        for key in depth_keys:
            if key in real_obs:
                if key not in depth_batches:
                    depth_batches[key] = []
                depth_batches[key].append((episode_idx, real_obs[key]))
    
    # Initialize result list
    for _ in range(len(real_obs_list)):
        policy_obs_list.append({})
    
    # Process RGB batches
    for key, episode_data_list in rgb_batches.items():
        # Stack all episodes for this key
        all_rgb_data = []
        episode_indices = []
        
        for episode_idx, rgb_data in episode_data_list:
            if rgb_data.ndim == 4 and rgb_data.shape[-1] == 3:
                processed_rgb = rgb_data.transpose(0, 3, 1, 2)  # (N, H, W, 3) -> (N, 3, H, W)
            elif rgb_data.ndim == 3 and rgb_data.shape[-1] == 3:
                processed_rgb = rgb_data.transpose(2, 0, 1)     # (H, W, 3) -> (3, H, W)
            
            all_rgb_data.append(processed_rgb)
            episode_indices.append(episode_idx)
        
        # Batch process all RGB data for this key
        if all_rgb_data:
            stacked_rgb = np.concatenate(all_rgb_data, axis=0)
            rgb_tensor = torch.from_numpy(stacked_rgb.astype(np.float32))
            transformed_rgb = rgb_transform(rgb_tensor)
            transformed_rgb_np = transformed_rgb.numpy()
            
            # Split back to episodes
            current_idx = 0
            for i, (episode_idx, original_data) in enumerate(episode_data_list):
                episode_length = all_rgb_data[i].shape[0]
                episode_rgb = transformed_rgb_np[current_idx:current_idx + episode_length]
                policy_obs_list[episode_idx][key] = episode_rgb
                current_idx += episode_length
    
    # Process depth batches similarly
    for key, episode_data_list in depth_batches.items():
        all_depth_data = []
        episode_indices = []
        original_shapes = []
        
        for episode_idx, depth_data in episode_data_list:
            depth_normalized = (depth_data / 1000.0).astype(np.float32)
            all_depth_data.append(depth_normalized)
            episode_indices.append(episode_idx)
            original_shapes.append(depth_data.shape)
        
        if all_depth_data:
            stacked_depth = np.concatenate([d.reshape(-1, *d.shape[-2:]) for d in all_depth_data], axis=0)
            depth_tensor = torch.from_numpy(stacked_depth)
            transformed_depth = depth_transform(depth_tensor)
            transformed_depth_np = transformed_depth.numpy()
            
            # Split back to episodes with proper reshaping
            current_idx = 0
            for i, (episode_idx, original_data) in enumerate(episode_data_list):
                original_shape = original_shapes[i]
                episode_length = original_data.size // (original_data.shape[-2] * original_data.shape[-1])
                episode_depth = transformed_depth_np[current_idx:current_idx + episode_length]
                
                if original_shape[0] > 1:  # 3D case
                    episode_depth = np.expand_dims(episode_depth, axis=1)  # (N, H, W) -> (N, 1, H, W)
                elif len(original_shape) == 2:  # 2D case
                    episode_depth = np.expand_dims(episode_depth, axis=0)  # (H, W) -> (1, H, W)
                
                policy_obs_list[episode_idx][key] = episode_depth
                current_idx += episode_length
    
    # Process remaining keys (non-image data) episode by episode
    for episode_idx, real_obs in enumerate(real_obs_list):
        for key, value in real_obs.items():
            if key not in rgb_keys and key not in depth_keys:
                if key == 'robot_eef_pose':
                    trans = value[..., 0:3]
                    rot = value[..., 3:6]
                    gripper = value[..., 6:]
                    new_rot = axis_angle_to_6drot.forward(rot)
                    new_eef = np.concatenate([trans, new_rot, gripper], axis=-1)
                    policy_obs_list[episode_idx][key] = value
                    policy_obs_list[episode_idx]['robot_eef_9dpose'] = new_eef
                else:
                    policy_obs_list[episode_idx][key] = value
    
    return policy_obs_list
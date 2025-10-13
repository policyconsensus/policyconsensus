import einops
from typing import Optional
import time

import numpy as np
import h5py
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm
# from modular_policy.common.pcd_utils import (aggr_point_cloud_from_data, remove_plane_from_mesh,
#     visualize_pcd)
from modular_policy.common.trans_utils import transform_to_world, transform_from_world, interpolate_poses
from modular_policy.model.common.rotation_transformer import RotationTransformer
import cProfile
import pstats
import io
import os
import sys
from contextlib import contextmanager
import dask.array as da
from dask import delayed
import dask.bag as db
from scipy.spatial.transform import Rotation as R, Slerp

# from modular_policy.common.view_util import generate_n_cam_to_world_matrices, extract_rpy_distance_range
# from modular_policy.common.cv2_util import get_image_transform, visualize_pca_features, ImgNorm_inv, load_images_mast3r, plot_affordance_overlay

# from mast3r.model import AsymmetricMASt3R
# from mast3r.fast_nn import fast_reciprocal_NNs
# from dust3r.inference import inference
# from dust3r.utils.image import load_images
# import starster

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter


@contextmanager
def suppress_output():
    """
    Suppresses stdout and stderr, including tqdm outputs, within the context.
    """
    dummy = open(os.devnull, 'w')
    original_stdout, original_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = dummy, dummy
        yield  # Run the code within this block with stdout/stderr suppressed
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
        dummy.close()

def save_dict_to_hdf5(dic, config_dict, filename, attr_dict=None):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        if attr_dict is not None:
            for key, item in attr_dict.items():
                h5file.attrs[key] = item
        recursively_save_dict_contents_to_group(h5file, '/', dic, config_dict)

def recursively_save_dict_contents_to_group(h5file, path, dic, config_dict):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray):
            # h5file[path + key] = item
            if key not in config_dict:
                config_dict[key] = {}
            dset = h5file.create_dataset(path + key, shape=item.shape, **config_dict[key])
            dset[...] = item
        elif isinstance(item, dict):
            if key not in config_dict:
                config_dict[key] = {}
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item, config_dict[key])
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    # with h5py.File(filename, 'r') as h5file:
    #     return recursively_load_dict_contents_from_group(h5file, '/')
    h5file = h5py.File(filename, 'r')
    return recursively_load_dict_contents_from_group(h5file, '/'), h5file

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            # ans[key] = np.array(item)
            ans[key] = item
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def modify_hdf5_from_dict(filename, dic):
    """
    Modify hdf5 file from a dictionary
    """
    with h5py.File(filename, 'r+') as h5file:
        recursively_modify_hdf5_from_dict(h5file, '/', dic)

def recursively_modify_hdf5_from_dict(h5file, path, dic):
    """
    Modify hdf5 file from a dictionary recursively
    """
    for key, item in dic.items():
        if isinstance(item, np.ndarray) and key in h5file[path]:
            h5file[path + key][...] = item
        elif isinstance(item, dict):
            recursively_modify_hdf5_from_dict(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot modify %s type'%type(item))

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
    # vis_post_actions(actions[:,10:])
    return actions

def _homo_to_9d_action(raw_actions, rotation_transformer=None):
    if rotation_transformer is None:
        rotation_transformer = RotationTransformer(
            from_rep='matrix', to_rep='rotation_6d')

    trans = raw_actions[:, :3, 3]  # Extract translation vectors
    rot_mat = raw_actions[:, :3, :3]    # Extract rotation matrices
    rot_vec = rotation_transformer.forward(rot_mat)  # Convert rotation matrices to rotation vectors
    pose_vec = np.concatenate([
        trans, rot_vec
    ], axis=-1).astype(np.float32)
    return pose_vec

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

def _6d_axis_angle_to_homo(raw_actions):
    trans = raw_actions[:, :3]  
    axis_angle = raw_actions[:, 3:]  
    rot_mat = convert_axis_angle_to_matrix(axis_angle)
    
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
    # vis_post_actions(actions[:,10:])
    return actions

def convert_rotation_6d_to_matrix(rot_6d):
    rotation_transformer = RotationTransformer(
        from_rep='rotation_6d', to_rep='matrix')
    rot_mat = rotation_transformer.forward(rot_6d)
    return rot_mat

def convert_matrix_to_rotation_6d(rot_mat):
    rotation_transformer = RotationTransformer(
        from_rep='matrix', to_rep='rotation_6d')
    rot_6d = rotation_transformer.forward(rot_mat)
    return rot_6d

def convert_axis_angle_to_rotation_6d(axis_angle):
    rotation_transformer = RotationTransformer(
        from_rep='axis_angle', to_rep='rotation_6d')
    rot_6d = rotation_transformer.forward(axis_angle)
    return rot_6d

def convert_rotation_6d_to_axis_angle(rot_6d):
    rotation_transformer = RotationTransformer(
        from_rep='rotation_6d', to_rep='axis_angle')
    axis_angle = rotation_transformer.forward(rot_6d)
    return axis_angle

def convert_axis_angle_to_matrix(axis_angle):
    rotation_transformer = RotationTransformer(
        from_rep='axis_angle', to_rep='matrix')
    rot_mat = rotation_transformer.forward(axis_angle)
    return rot_mat

def convert_matrix_to_axis_angle(matrix):
    rotation_transformer = RotationTransformer(
        from_rep='matrix', to_rep='axis_angle')
    axis_angle = rotation_transformer.forward(matrix)
    return axis_angle
# def compute_relative_pose(pos1, rot1, pos2, rot2):
#     # Relative position
#     rel_pos = pos2 - pos1
    
#     # Convert 6D rotation to rotation matrix
#     rot_matrix1 = convert_rotation_6d_to_matrix(rot1)
#     rot_matrix2 = convert_rotation_6d_to_matrix(rot2)
    
#     # Compute relative rotation
#     if len(rot_matrix1.shape) == 3:  # Batch operation
#         rel_rot_matrix = np.matmul(rot_matrix1.transpose(0, 2, 1), rot_matrix2)
#     else:  # Single operation (2D case)
#         rel_rot_matrix = np.matmul(rot_matrix1.T, rot_matrix2)
    
#     # Convert relative rotation back to 6D representation
#     rel_rot_6d = convert_matrix_to_rotation_6d(rel_rot_matrix)
    
#     return rel_pos, rel_rot_6d

# def apply_relative_pose(pos1, rot1, rel_pos, rel_rot):
#     # Apply relative position
#     new_pos = pos1 + rel_pos
    
#     # Convert 6D rotation to rotation matrix
#     rot_matrix1 = convert_rotation_6d_to_matrix(rot1)
#     rel_rot_matrix = convert_rotation_6d_to_matrix(rel_rot)
    
#     # Apply relative rotation
#     new_rot_matrix = np.matmul(rot_matrix1, rel_rot_matrix)
    
#     # Convert new rotation back to 6D representation
#     new_rot_6d = convert_matrix_to_rotation_6d(new_rot_matrix)
    
#     return new_pos, new_rot_6d

def policy_action_to_env_action(raw_actions, cur_eef_pose_6d=None, init_tool_pose_9d=None, 
                                tool_in_eef=None,action_mode='eef', 
                                use_tool_in_base=False,
                                robot_base_pose_in_world=None, cam_extrinsic=None, 
                                num_bots=1, debug=False):
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
    elif action_mode == 'eef':
        if init_tool_pose_9d is not None or use_tool_in_base:            
            pred_eef_pose = []
            if debug: 
                eef_in_base_ls = []
            
            if use_tool_in_base:
                tool_in_base = _9d_to_homo_action(raw_actions)
                pred_eef_pose = tool_in_base@np.linalg.inv(tool_in_eef) 
            else:
                eef_in_base = _6d_axis_angle_to_homo(cur_eef_pose_6d[None,:])[0]

                init_tool_in_cam = _9d_to_homo_action(init_tool_pose_9d[None,:])[0]
                tool_in_cam = _9d_to_homo_action(raw_actions)
                pred_eef_pose = eef_in_base @ tool_in_eef @ (np.linalg.inv(init_tool_in_cam) @ tool_in_cam)@np.linalg.inv(tool_in_eef) 
                
            pred_eef_pose_6d = _homo_to_6d_axis_angle(pred_eef_pose)

            if debug:
                for pred_tool_pose in tool_in_cam:

                    new_eef_in_base = eef_in_base @ tool_in_eef @ (np.linalg.inv(init_tool_in_cam) @ pred_tool_pose)@np.linalg.inv(tool_in_eef)
                    pred_eef_pose.append(_homo_to_6d_axis_angle(new_eef_in_base[None,:])[0])
                    
                    eef_in_base_debug = np.linalg.inv(robot_base_pose_in_world)@cam_extrinsic@pred_tool_pose@np.linalg.inv(tool_in_eef)
                    eef_in_base_ls.append(_homo_to_6d_axis_angle(eef_in_base_debug[None,:])[0])
                    assert np.allclose(new_eef_in_base, eef_in_base_debug, atol=1e-7)
                pred_eef_pose_6d = np.array(pred_eef_pose)

            if debug:
                tool_in_base_ls = np.array(tool_in_base_ls)
            return pred_eef_pose_6d # eef_action_to_6d(pred_eef_pose)
        else:
            return eef_action_to_6d(raw_actions)
        
    elif action_mode == 'joint':
        return raw_actions
    
def point_cloud_proc(shape_meta, color_seq, depth_seq, extri_seq, intri_seq,
                  robot_base_pose_in_world_seq = None, qpos_seq=None, expected_labels=None, 
                  exclude_threshold=0.01, exclude_colors=[], teleop_robot=None, tool_names=[None], profile=False):
    # shape_meta: (dict) shape meta data for d3fields
    # color_seq: (np.ndarray) (T, V, H, W, C)
    # depth_seq: (np.ndarray) (T, V, H, W)
    # extri_seq: (np.ndarray) (T, V, 4, 4)
    # intri_seq: (np.ndarray) (T, V, 3, 3)
    # robot_name: (str) name of robot
    # meshes: (list) list of meshes
    # offsets: (list) list of offsets
    # finger_poses: (dict) dict of finger poses, mapping from finger name to (T, 6)
    # expected_labels: (list) list of expected labels
    if profile:
        # Initialize profiler
        pr = cProfile.Profile()
        pr.enable()


    boundaries = shape_meta['info']['boundaries']
    if boundaries == 'none':
        boundaries = None
    N_total = shape_meta['shape'][1]
    max_pts_num = shape_meta['shape'][1]
    
    resize_ratio = shape_meta['info']['resize_ratio']
    reference_frame = shape_meta['info']['reference_frame'] if 'reference_frame' in shape_meta['info'] else 'world'
    
    num_bots = 2 if robot_base_pose_in_world_seq.shape[2] == 8 else 1
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.reshape(robot_base_pose_in_world_seq.shape[0], 4, num_bots, 4)
    robot_base_pose_in_world_seq = robot_base_pose_in_world_seq.transpose(0, 2, 1, 3)
    
    # TODO: pay attention to this part 
    # robot_base_in_world_temp = np.array([[[1.0, 0, 0.0, 0.80],
    #                                 [0, 1.0, 0.0, -0.22],
    #                                 [0.0, 0.0, 1.0, 0.03],
    #                                 [0.0, 0.0, 0.0, 1.0]],
    #                                 [[1.0, 0, 0.0, -0.515],
    #                                 [0, 1.0, 0.0, -0.22],
    #                                 [0.0, 0.0, 1.0, 0.03],
    #                                 [0.0, 0.0, 0.0, 1.0]],])
    # robot_base_pose_in_world_seq = np.repeat(robot_base_in_world_temp[None, ...], robot_base_pose_in_world_seq.shape[0], axis=0)

    H, W = color_seq.shape[2:4]
    resize_H = int(H * resize_ratio)
    resize_W = int(W * resize_ratio)
    
    new_color_seq = np.zeros((color_seq.shape[0], color_seq.shape[1], resize_H, resize_W, color_seq.shape[-1]), dtype=np.uint8)
    new_depth_seq = np.zeros((depth_seq.shape[0], depth_seq.shape[1], resize_H, resize_W), dtype=np.float32)
    new_intri_seq = np.zeros((intri_seq.shape[0], intri_seq.shape[1], 3, 3), dtype=np.float32)
    for t in range(color_seq.shape[0]):
        for v in range(color_seq.shape[1]):
            new_color_seq[t,v] = cv2.resize(color_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_depth_seq[t,v] = cv2.resize(depth_seq[t,v], (resize_W, resize_H), interpolation=cv2.INTER_NEAREST)
            new_intri_seq[t,v] = intri_seq[t,v] * resize_ratio
            new_intri_seq[t,v,2,2] = 1.
    color_seq = new_color_seq
    depth_seq = new_depth_seq
    intri_seq = new_intri_seq
    T, V, H, W, C = color_seq.shape
    # assert H == 240 and W == 320 and C == 3
    aggr_src_pts_ls = []
    
    use_robot_pcd = shape_meta['info']['rob_pcd']
    use_tool_pcd = shape_meta['info']['eef_pcd']
    N_per_link = shape_meta['info']['N_per_link']
    N_eef = shape_meta['info']['N_eef']
    N_joints = shape_meta['info']['N_joints']
    voxel_size = shape_meta['info'].get('voxel_size', 0.02)
    remove_plane = shape_meta['info'].get('remove_plane', False)
    plane_dist_thresh = shape_meta['info'].get('plane_dist_thresh', 0.01)

    # Assigning a bright magenta to the robot point cloud (robot_pcd)
    # RGB for a bright magenta: [1.0, 0, 1.0]
    robot_color = np.array([1.0, 0, 1.0])  # Bright Magenta

    # Assigning a fluorescent green to the end-effector point cloud (ee_pcd)
    # RGB for a fluorescent green: [0.5, 1.0, 0]
    ee_color = shape_meta['info'].get('ee_color', [0.5, 1.0, 0])  # Fluorescent Green
    # ee_color = np.array([0.5, 1.0, 0])  # Fluorescent Green

    for t in range(T):
        curr_qpos = qpos_seq[t]
        qpos_dim = curr_qpos.shape[0] // num_bots
        
        robot_pcd_ls = []
        ee_pcd_ls = []
        

        for rob_i in range(num_bots):
            # transform robot pcd to world frame    
            robot_base_pose_in_world = robot_base_pose_in_world_seq[t, rob_i] if robot_base_pose_in_world_seq is not None else None

            tool_name = tool_names[rob_i]
            # compute robot pcd
            if use_robot_pcd:
                robot_pcd_pts = teleop_robot.get_robot_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], N_joints=N_joints, N_per_link=N_per_link, out_o3d=False)
                robot_pcd_pts = (robot_base_pose_in_world @ np.concatenate([robot_pcd_pts, np.ones((robot_pcd_pts.shape[0], 1))], axis=-1).T).T[:, :3]

                robot_pcd_color = np.tile(robot_color, (robot_pcd_pts.shape[0], 1))
                robot_pcd_pts = np.concatenate([robot_pcd_pts, robot_pcd_color], axis=-1)
                robot_pcd_ls.append(robot_pcd_pts)

            if use_tool_pcd:
                ee_pcd_pts = teleop_robot.get_tool_pcd(curr_qpos[qpos_dim*rob_i:qpos_dim*(rob_i+1)], tool_name, N_per_inst=N_eef)    
                ee_pcd_pts = (robot_base_pose_in_world @ np.concatenate([ee_pcd_pts, np.ones((ee_pcd_pts.shape[0], 1))], axis=-1).T).T[:, :3]
            
                ee_pcd_color = np.tile(ee_color, (ee_pcd_pts.shape[0], 1))
                ee_pcd_pts = np.concatenate([ee_pcd_pts, ee_pcd_color], axis=-1)
                ee_pcd_ls.append(ee_pcd_pts)
                
        robot_pcd = np.concatenate(robot_pcd_ls , axis=0) if use_robot_pcd else None
        ee_pcd = np.concatenate(ee_pcd_ls, axis=0) if use_tool_pcd  else None

        aggr_src_pts = aggr_point_cloud_from_data(color_seq[t], depth_seq[t], intri_seq[t], extri_seq[t], 
                                                  downsample=True, N_total=N_total,
                                                  boundaries=boundaries, 
                                                  out_o3d=False, 
                                                  exclude_colors=exclude_colors,
                                                  voxel_size=voxel_size,
                                                  remove_plane=remove_plane,
                                                  plane_dist_thresh=plane_dist_thresh,)
        aggr_src_pts = np.concatenate(aggr_src_pts, axis=-1) # (N_total, 6), 6: (x, y, z, r, g, b)
        aggr_src_pts = np.concatenate([aggr_src_pts, robot_pcd], axis=0) if use_robot_pcd else aggr_src_pts
        aggr_src_pts = np.concatenate([aggr_src_pts, ee_pcd], axis=0) if use_tool_pcd else aggr_src_pts
        
        # transform to reference frame
        if reference_frame == 'world':
            pass
        elif reference_frame == 'robot':
            transformed_xyz = (np.linalg.inv(robot_base_pose_in_world_seq[t, 0]) @ np.concatenate([aggr_src_pts[:, :3], np.ones((aggr_src_pts.shape[0], 1))], axis=-1).T).T[:, :3]
            aggr_src_pts = np.concatenate([transformed_xyz, aggr_src_pts[:, 3:]], axis=1)
            
        aggr_src_pts_ls.append(aggr_src_pts.astype(np.float32))
            
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = 'cumulative' # or 'time'
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(10)
            print(s.getvalue())
 
    
    return aggr_src_pts_ls

def augment_with_novel_views(t, color_seq, mast3r_size, starster_mast3r_model, dummy_file_list, device, feature_view_num):

    src_img_list = (load_images_mast3r)(color_seq[t], size=mast3r_size, verbose=False, device='cpu')
    imgs = ([img_info['img'][0] for img_info in src_img_list])
    
    # Suppress output during scene reconstruction and make it a delayed operation
    with suppress_output():
        scene = (starster.reconstruct_scene)(starster_mast3r_model, imgs, dummy_file_list, device=device)

    gs = (starster.GSTrainer)(scene, device=device)
    
    random_indices = torch.randint(low=0, high=feature_view_num, size=(1,), device=device)
    selected_intrinsics = (gs.scene.intrinsics())[random_indices]
    
    roll_range, pitch_range, yaw_range, distance_range = extract_rpy_distance_range(torch.inverse(gs.scene.w2c()))

    # Delay camera-to-world matrix generation and inverse computation
    c2w_matrices = (generate_n_cam_to_world_matrices)(1, roll_range, pitch_range, yaw_range, distance_range, output_format='torch').to(device)
    w2c_matrices = (torch.inverse)(c2w_matrices)
    
    # Delay optimization and rendering as well
    gs.run_optimization(100, enable_pruning=True, verbose=False)
    gs.run_optimization(500, enable_pruning=False, verbose=False)

    aug_img, alpha, info = (gs.render_views)(w2c_matrices, selected_intrinsics, mast3r_size, mast3r_size)
    
    # Process augmented images as delayed objects
    aug_img_info = (load_images_mast3r)(aug_img.permute(0, 3, 1, 2), size=mast3r_size, verbose=False)
    
    src_img_list[0]['img'] = (src_img_list[0]['img'].to)(device)
    mast3r_imgs = [src_img_list[0]] + aug_img_info

    return mast3r_imgs

def generate_novel_view(gs, aug_img_num, new_w2c, device, mast3r_size):
    
    random_indices = torch.randint(low=0, high=2, size=(aug_img_num,), device=device)
    selected_intrinsics = (gs.scene.intrinsics())[random_indices]

    # c2w_matrices = (generate_n_cam_to_world_matrices)(1, roll_range, pitch_range, yaw_range, distance_range, output_format='torch').to(device)
    # w2c_matrices = (torch.inverse)(c2w_matrices)

    render_view_arg = {'w2c': new_w2c, 
                       'intrinsics': selected_intrinsics,
                       'width': mast3r_size, 
                       'height': mast3r_size}
    
    aug_img, alpha, info = gs.render_views(**render_view_arg)

    return aug_img


def interpolate_homogeneous_matrix(w2c1, w2c2, alpha):
    """
    Interpolates two homogeneous transformation matrices w2c1 and w2c2.

    Args:
        w2c1: First homogeneous matrix (shape [4, 4]).
        w2c2: Second homogeneous matrix (shape [4, 4]).
        alpha: Interpolation factor (0.0 to 1.0).

    Returns:
        Interpolated homogeneous matrix (shape [4, 4]).
    """
    # Extract rotation and translation
    R1, t1 = w2c1[:3, :3], w2c1[:3, 3]
    R2, t2 = w2c2[:3, :3], w2c2[:3, 3]

    # Convert rotation matrices to quaternions
    q1 = R.from_matrix(R1.cpu().numpy()).as_quat()
    q2 = R.from_matrix(R2.cpu().numpy()).as_quat()

    # Interpolate rotations using Slerp
    key_times = [0, 1]  # Start and end times for Slerp
    key_rotations = R.from_quat([q1, q2])
    slerp = Slerp(key_times, key_rotations)
    interpolated_rotation = slerp([alpha])[0]  # Interpolate for given alpha
    R_interpolated = torch.tensor(interpolated_rotation.as_matrix(), device=w2c1.device)

    # Interpolate translation
    t_interpolated = (1 - alpha) * t1 + alpha * t2

    # Construct the interpolated homogeneous matrix
    interpolated = torch.eye(4, device=w2c1.device)
    interpolated[:3, :3] = R_interpolated
    interpolated[:3, 3] = t_interpolated

    return interpolated

def generate_intermediate_matrices(
    w2c1, w2c2, num_matrices=6, random_matrices=False
):
    """
    Generates w2c matrices with a mix of those close to w2c1, w2c2, and optionally random matrices.

    Args:
        w2c1: First w2c matrix (shape [4, 4]).
        w2c2: Second w2c matrix (shape [4, 4]).
        num_matrices: Total number of matrices to generate (should be even if no random_matrices).
        random_matrices: If True, generates some matrices with fully random interpolation.

    Returns:
        Tensor of generated w2c matrices (shape [num_matrices, 4, 4]).
    """
    if not random_matrices and num_matrices % 2 != 0:
        raise ValueError("num_matrices must be even when random_matrices is False to split evenly between w2c1 and w2c2")

    generated_matrices = []

    # Determine split for w2c1 and w2c2 biased matrices
    if random_matrices:
        num_close = num_matrices // 2  # Half for biased matrices
        num_random = num_matrices - num_close  # Remaining for random matrices
    else:
        num_close = num_matrices // 2
        num_random = 0

    # Generate alphas biased towards w2c1
    alphas_w2c1 = torch.distributions.Beta(5, 2).sample([num_close]).tolist()

    # Generate alphas biased towards w2c2
    alphas_w2c2 = torch.distributions.Beta(2, 5).sample([num_close]).tolist()

    # Generate matrices close to w2c1
    for alpha in alphas_w2c1:
        interpolated = interpolate_homogeneous_matrix(w2c1, w2c2, alpha)
        generated_matrices.append(interpolated)

    # Generate matrices close to w2c2
    for alpha in alphas_w2c2:
        interpolated = interpolate_homogeneous_matrix(w2c1, w2c2, alpha)
        generated_matrices.append(interpolated)

    # Generate fully random matrices (random interpolation)
    for _ in range(num_random):
        alpha = torch.rand(1).item()  # Uniform random interpolation
        interpolated = interpolate_homogeneous_matrix(w2c1, w2c2, alpha)
        generated_matrices.append(interpolated)

    # Stack all generated matrices into a tensor
    return torch.stack(generated_matrices)

def setup_scene_and_gs(color_seq, mast3r_size, starster_mast3r_model, dummy_file_list, aug_img_num, device):

    src_img_list = (load_images_mast3r)(color_seq[0], size=mast3r_size, verbose=False, device='cpu')
    imgs = ([img_info['img'][0] for img_info in src_img_list])
    
    # Suppress output during scene reconstruction and make it a delayed operation
    with suppress_output():
        scene = (starster.reconstruct_scene)(starster_mast3r_model, imgs, dummy_file_list, device=device)

    gs = (starster.GSTrainer)(scene, device=device)
    
    # roll_range, pitch_range, yaw_range, distance_range = extract_rpy_distance_range(torch.inverse(gs.scene.w2c()))
    new_matrices = generate_intermediate_matrices(gs.scene.w2c()[0], gs.scene.w2c()[1], num_matrices=aug_img_num, random_matrices=False)

    # Delay optimization and rendering as well
    gs.run_optimization(5000, enable_pruning=True, verbose=False)
    gs.run_optimization(100, enable_pruning=False, verbose=False)
    
    src_img_list[0]['img'] = (src_img_list[0]['img'].to)(device)


    return gs, src_img_list, new_matrices

def generate_correspondence(desc1, desc2, img1, img2, device='cuda', n_viz=20, features=None, output_format='affordance', 
                            normalization_method='min-max', rm_irrel=False, target_size=(224,224), vis=False):
    """
    Generate correspondences between two images given their descriptors.

    Parameters:
    - desc1: Tensor of descriptors for the first image
    - desc2: Tensor of descriptors for the second image
    - img1: Tensor of the first image (in the form (C, H, W))
    - img2: Tensor of the second image (in the form (C, H, W))
    - device: The device to run the calculations on (default is 'cuda')
    - n_viz: Number of correspondences to visualize (default is 20)
    - output_format: afforance or mask
    - rm_irrel: If True, removes areas with affordance == 0

    Returns:
    - matches_im0: Matched keypoints in the first image
    - matches_im1: Matched keypoints in the second image
    - Visualization of matches between the two images
    """
    # Find 2D-2D matches between the two images using descriptors
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=8,
                                                   device=device, dist='dot', block_size=2**13)

    # Ignore small border around the edge
    H0, W0 = img1.shape[-2:]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = img2.shape[-2:]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    # Filter valid matches
    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

    x0, y0 = matches_im0[:, 0], matches_im0[:, 1]
    x1, y1 = matches_im1[:, 0], matches_im1[:, 1]
    if output_format == 'mask':
        
        mask1 = torch.zeros((H0, W0), dtype=torch.uint8, device=device)
        mask2 = torch.zeros((H1, W1), dtype=torch.uint8, device=device)
        mask1[y0, x0] = 1
        mask2[y1, x1] = 1
        overlay1, overlay2 = mask1.cpu().numpy(), mask2.cpu().numpy()

    elif output_format == 'affordance':
        # Generate affordance maps (heatmaps)
        affordance1 = torch.zeros((H0, W0), dtype=torch.float32, device=device)
        affordance2 = torch.zeros((H1, W1), dtype=torch.float32, device=device)

        affordance1[y0, x0] += 1.0  # Increment at keypoints
        affordance2[y1, x1] += 1.0

        # Smooth the affordance map for better visualization
        affordance1 = gaussian_filter(affordance1.cpu().numpy(), sigma=4)
        affordance2 = gaussian_filter(affordance2.cpu().numpy(), sigma=4)
        
        # Normalize the affordance maps
        if normalization_method == 'min-max':
            affordance1 = affordance1 / (affordance1.max() + 1e-8)
            affordance2 = affordance2 / (affordance2.max() + 1e-8)
        elif normalization_method == 'z-score':
            affordance1 = (affordance1 - affordance1.mean()) / (affordance1.std() + 1e-8)
            affordance2 = (affordance2 - affordance2.mean()) / (affordance2.std() + 1e-8)
            
        if rm_irrel:
            # Ensure affordance maps are in the correct shape [H, W]
            mask1 = torch.tensor(affordance1 > 0, dtype=torch.bool, device=img1.device)  # [H, W]
            mask2 = torch.tensor(affordance2 > 0, dtype=torch.bool, device=img2.device)  # [H, W]

            # Expand the mask along the channel dimension to match image shape [C, H, W]
            mask1 = mask1[None,None,...].expand(-1, img1.shape[1], -1, -1)
            mask2 = mask2[None,None,...].expand(-1, img2.shape[1], -1, -1)

            # Set background regions to -1 where the affordance is 0
            img1 = img1.masked_fill(~mask1, -1)
            img2 = img2.masked_fill(~mask2, -1)
        overlay1, overlay2 = affordance1, affordance2
    else:
        raise ValueError('Invalid output format')
    
    # Extract the corresponding descriptors from desc1 and desc2 using the x, y coordinates
    desc_matches_im0 = desc1[y0, x0]  # Corresponding descriptors from the first image
    if features is not None:
        desc_matches_im1 = features[y1, x1]  # Corresponding descriptors from the second image
    else:    
        desc_matches_im1 = desc2[y1, x1]  # Corresponding descriptors from the second image

    # Visualize a few matches
    num_matches = matches_im0.shape[0]
    match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]
    
    def denormalize_image(img):
        mean = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=img.device).view(1, 3, 1, 1)
        return (img * std + mean)
    
    def resize_rgb_and_affordance(rgb_img, affordance, new_size):
        """Resize RGB channels and affordance map separately, then concatenate."""
        # Ensure RGB image is [1, 3, H, W]
        if rgb_img.ndim == 3:
            rgb_img = rgb_img.unsqueeze(0)

        # Resize RGB image with bilinear interpolation
        rgb_resized = F.interpolate(
            rgb_img, size=new_size, mode='area'
        )
        if affordance != None:
            # Ensure affordance map is [1, 1, H, W]
            if affordance.ndim == 2:
                affordance = affordance.unsqueeze(0).unsqueeze(0)

            # Resize affordance map with nearest-neighbor interpolation
            affordance_resized = F.interpolate(
                affordance, size=new_size, mode='bicubic'
            )

            # Concatenate RGB and affordance map along the channel dimension
            img_with_affordance = torch.cat((rgb_resized, affordance_resized), dim=1)  # [1, 4, new_height, new_width]

            return img_with_affordance.permute(0, 2, 3, 1)
        else:
            return rgb_resized.permute(0, 2, 3, 1)

    ret_img1 = resize_rgb_and_affordance(denormalize_image(img1), None if rm_irrel else torch.from_numpy(affordance1).to(img1.device), target_size).cpu().numpy()
    ret_img2 = resize_rgb_and_affordance(denormalize_image(img2), None if rm_irrel else torch.from_numpy(affordance2).to(img2.device), target_size).cpu().numpy()


    if vis:
        # Normalize overlays for visualization
        overlay1 = overlay1 / (overlay1.max() + 1e-8)  # Avoid division by zero
        overlay2 = overlay2 / (overlay2.max() + 1e-8)

        image_mean = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)
        image_std = torch.as_tensor([0.5, 0.5, 0.5], device='cpu').reshape(1, 3, 1, 1)

        # Prepare images for visualization
        viz_imgs = []
        for i, img in enumerate([img1, img2]):
            rgb_tensor = img * image_std + image_mean
            img_np = rgb_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

            overlay = overlay1 if i == 0 else overlay2
            cmap = plt.get_cmap('viridis')
            overlay_rgb = cmap(overlay)[:, :, :3]  
            
            blended_img = np.clip(img_np * 0.6 + overlay_rgb * 0.4, 0, 1)
            viz_imgs.append(blended_img)

        H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
        vis_img1 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        vis_img2 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((vis_img1, vis_img2), axis=1)

        # Plot the matches
        plt.figure()
        plt.imshow(img)
        cmap = plt.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        plt.show(block=True)

    return (viz_matches_im0, viz_matches_im1), (ret_img1, ret_img2)


def feature_proc(color_seq, feature_raw_image, 
                 aug_img_num, shape_meta, 
                 aug_mult, updated_episode_length, 
                 mast3r_size, starster_mast3r_model, 
                 device, mast3r_model, mast3r_batch, 
                 reference_imgs, num_matches,
                 novel_view, affordance_norm, rm_irrel):
    # if novel_view:
    dummy_file_list = [f"dummy_{i}.png" for i in range(2)]
    target_size = (mast3r_size, mast3r_size)

    c,h,w = tuple(shape_meta['obs']['camera_features']['shape'])
    # if shape_meta['obs']['camera_features']['type'] == 'sparse_feature':
    #     episode_views_features = np.empty((aug_mult, updated_episode_length, sparse_feature_num, c), dtype=np.float32)
    # else:
    episode_views_features = np.empty((aug_mult, updated_episode_length, h, w, c), dtype=np.float32)
    

    for batch_start in range(0, len(color_seq)):
        batch_end = min(batch_start + 1, len(color_seq))
        
        color_seq_batch = color_seq[batch_start:batch_end]
        if novel_view:
            gs, src_img_list, new_w2c = setup_scene_and_gs(color_seq_batch, mast3r_size, starster_mast3r_model, dummy_file_list, aug_img_num-1, device) 
        
        correspondence_succeeded = False
        use_other_view = False
        while not correspondence_succeeded:
            try:
                if novel_view: 
                    if use_other_view == False:
                        aug_img = generate_novel_view(gs, aug_img_num-1, new_w2c, device, mast3r_size)
                        aug_img_info = (load_images_mast3r)(aug_img.permute(0, 3, 1, 2), size=mast3r_size, verbose=False)
                        pair_img_list = [src_img_list[0]] + aug_img_info + [src_img_list[1]]
                    else:
                        pair_img_list = [src_img_list[0]] + [src_img_list[1]] * aug_img_num
                else:
                    src_img_list = load_images_mast3r(color_seq_batch[0], size=mast3r_size, verbose=False)
                    if len(src_img_list) == 1:
                        pair_img_list = [src_img_list[0]] * (aug_img_num+1)
                    else:
                        pair_img_list = [src_img_list[0]] + [src_img_list[1]] * aug_img_num
                    
                if feature_raw_image:
                    episode_views_features[:, batch_start:batch_end] = np.concatenate([
                        F.interpolate(ImgNorm_inv(img_info['img']).detach(), size=(h,w), mode='area').permute(0, 2, 3, 1).cpu().numpy() 
                        for img_info in pair_img_list
                    ])[:,None,...]

                else:
                    delayed_pairs = [
                        delayed((ref_img, single_img))
                        for single_img in pair_img_list
                        for ref_img in reference_imgs
                    ]
                    # features = [feature_extractor.get_feature_representation(ImgNorm_inv(img_data['img']), target_size) for img_data in pair_img_list]

                    delayed_outputs = [
                        delayed(inference)([mast3r_pairs], mast3r_model, device, batch_size=mast3r_batch, verbose=False)
                        for mast3r_pairs in delayed_pairs
                    ]
                    delayed_descriptors = [delayed(generate_correspondence)(
                        output['pred1']['desc'][0], 
                        output['pred2']['desc'][0], 
                        output['view1']['img'], 
                        output['view2']['img'], 
                        device='cuda', n_viz=num_matches,
                        target_size=(h,w),
                        normalization_method=affordance_norm, rm_irrel=rm_irrel)[1][1] for output in delayed_outputs]

                    descriptors_computed = np.concatenate([desc.compute() for desc in delayed_descriptors])
                    episode_views_features[:, batch_start:batch_end] = descriptors_computed[:,None,...]
                correspondence_succeeded = True
            except Exception as e:
                use_other_view = True
                print(f"Error in generate_correspondence: {e}. Retrying with a new novel view...")
                continue   
    return np.concatenate(episode_views_features)


    # else:
    #     mast3r_imgs = delayed(load_images_mast3r)(color_seq, size=mast3r_size, verbose=False)
    #     # mast3r_imgs = load_images_mast3r(color_seq, size=mast3r_size, verbose=False)
    #     mast3r_imgs_bag = db.from_delayed(mast3r_imgs)
    #     del color_seq
        
    #     for ref_img in reference_imgs:
    #         # mast3r_pairs = [tuple([ref_img] + [single_img]) for single_img in mast3r_imgs]
    #         # mast3r_pairs = delayed([tuple([ref_img, single_img]) for single_img in mast3r_imgs])
    #         mast3r_pairs = mast3r_imgs_bag.map(lambda single_img: (ref_img, single_img))

    #         # mast3r_output = inference(mast3r_pairs, mast3r_model, device, batch_size=mast3r_batch, verbose=False)
    #         # mast3r_pred2 = mast3r_output['pred2']
    #         # descriptor =  mast3r_pred2['desc'].squeeze(0).detach().numpy().astype(np.float32)

    #         mast3r_output = delayed(inference)(mast3r_pairs, mast3r_model, device, batch_size=mast3r_batch, verbose=False)
    #         mast3r_pred2 = mast3r_output['pred2']
    #         descriptor = delayed(lambda pred: pred['desc'].squeeze(0).detach().numpy().astype(np.float32))(mast3r_pred2)

    #         # del mast3r_pairs
    #         # torch.cuda.empty_cache()
    #         # assert descriptor[0].shape == (h,w,c)
    #         # feature_data_dict[key].append(descriptor)
    #         # dask_descriptor = da.from_delayed(descriptor, shape=(updated_episode_length, h, w, c), dtype=np.float32)
    #     del mast3r_imgs
    #     return descriptor.computed()




class Dinov2FeatureExtractor:
    # references: https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb 
    # https://github.com/salihmarangoz/dinov2/blob/main/notebooks/matching_example/dinov2_matching.ipynb

    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vits14", smaller_edge_size=224, half_precision=False, device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda x: x if isinstance(x, torch.Tensor) else transforms.ToTensor()(x)),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet defaults
        ])

    # https://github.com/facebookresearch/dinov2/blob/255861375864acdd830f99fdae3d9db65623dafe/notebooks/features.ipynb
    def prepare_image(self, rgb_image):
        """
        Prepare the input image (tensor or numpy array) for feature extraction.

        Args:
        - rgb_image: torch.Tensor or np.ndarray
            The input image, either in numpy format (H, W, C) or as a tensor (C, H, W).

        Returns:
        - image_tensor: torch.Tensor
            The processed image tensor ready for feature extraction.
        - grid_size: tuple
            The grid size of the cropped image (H, W).
        - resize_scale: float
            The scale factor for resizing the image.
        """
        if isinstance(rgb_image, np.ndarray):
            # If input is numpy array (H, W, C), convert it to tensor and normalize to [0, 1]
            image_tensor = torch.tensor(rgb_image).permute(2, 0, 1).float() 
        elif isinstance(rgb_image, torch.Tensor):
            # If input is already a tensor, assume it's in (C, H, W) format
            image_tensor = rgb_image.float() 
        else:
            raise TypeError(f"Expected input to be np.ndarray or torch.Tensor, but got {type(rgb_image)}")
        
        if image_tensor.max()>10:
            image_tensor = image_tensor/ 255.0

        # Apply the transformation (assume self.transform handles tensors)
        image_tensor = self.transform(image_tensor)

        # Calculate the resize scale
        resize_scale = rgb_image.shape[3] / image_tensor.shape[3]

        # Get image dimensions (C, H, W)
        height, width = image_tensor.shape[2], image_tensor.shape[3]

        # Crop image dimensions to be multiples of the patch size
        cropped_width = width - (width % self.model.patch_size)
        cropped_height = height - (height % self.model.patch_size)

        # Crop the tensor to match the dimensions
        image_tensor = image_tensor[..., :cropped_height, :cropped_width]

        # Calculate the grid size (height and width divided by the patch size)
        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)

        return image_tensor, grid_size, resize_scale
  
    def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0]*self.model.patch_size*resize_scale), :int(grid_size[1]*self.model.patch_size*resize_scale)]
        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()
        return resized_mask
    
    def extract_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.half().to(self.device)
            else:
                image_batch = image_tensor.to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens
  
    def idx_to_source_position(self, idx, grid_size, resize_scale):
        row = (idx // grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        return row, col
  
    def get_feature_representation(self, image, target_size, resized_mask=None):
        image_tensor, grid_size, resize_scale = self.prepare_image(image)
        tokens = self.extract_features(image_tensor)

        if resized_mask is not None:
            tokens = tokens[resized_mask]
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = tokens.reshape((*grid_size, -1))
        reduced_tokens = reduced_tokens.permute(2, 0, 1).unsqueeze(0)  # Shape becomes [1, 384, 16, 16]
        upsampled_tokens = F.interpolate(reduced_tokens, size=target_size, mode='bilinear', align_corners=False)

        upsampled_tokens = upsampled_tokens.squeeze(0).permute(1, 2, 0)  
        return upsampled_tokens.cpu().numpy()


    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
        return normalized_tokens

  
    def get_combined_embedding_visualization(self, tokens1, token2, grid_size1, grid_size2, mask1=None, mask2=None, random_state=20):
        pca = PCA(n_components=3, random_state=random_state)
        
        token1_shape = tokens1.shape[0]
        if mask1 is not None:
            tokens1 = tokens1[mask1]
        if mask2 is not None:
            token2 = token2[mask2]
        combinedtokens= np.concatenate((tokens1, token2), axis=0)
        reduced_tokens = pca.fit_transform(combinedtokens.astype(np.float32))
        
        
        if mask1 is not None and mask2 is not None:
            resized_mask = np.concatenate((mask1, mask2), axis=0)
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        elif mask1 is not None and mask2 is None:
            return sys.exit("Either use both masks or none")
        elif mask1 is None and mask2 is not None:
            return sys.exit("Either use both masks or none")

        
        print("tokens1.shape", tokens1.shape)
        print("token2.shape", token2.shape)
        print("reduced_tokens.shape", reduced_tokens.shape)
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
            
        rgbimg1 = normalized_tokens[0:token1_shape,:]
        rgbimg2 = normalized_tokens[token1_shape:,:]
        
        rgbimg1 = rgbimg1.reshape((*grid_size1, -1))
        rgbimg2 = rgbimg2.reshape((*grid_size2, -1))
        return rgbimg1,rgbimg2


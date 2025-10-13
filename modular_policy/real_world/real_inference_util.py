import klampt
from typing import Dict, Callable, Tuple
import numpy as np
from einops import rearrange, reduce
from atm.common.cv2_util import get_image_transform, load_images_mast3r, visualize_pca_features, ImgNorm_inv

import os
from scipy.spatial.transform import Rotation as R
from atm.common.data_utils import _convert_actions, generate_correspondence
from atm.model.common.rotation_transformer import RotationTransformer
from atm.common.pcd_utils import update_intrinsics, convert_2d_to_3d

import matplotlib.pyplot as plt
from atm.model.common.tensor_util import unsqueeze
import torch
from pytorch3d.ops import efficient_pnp
from pytorch3d.transforms import matrix_to_rotation_6d
import dill
import torch.nn.functional as F

# from mast3r.model import AsymmetricMASt3R
# from mast3r.fast_nn import fast_reciprocal_NNs
# from dust3r.inference import inference
# from dust3r.utils.image import load_images

import cv2
import re
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../third_party'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../third_party/FoundationPose'))
from FoundationPose.estimater import *
from FoundationPose.datareader import *

def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        task_conf=None,
        kin_helper=None,
        real_obs_info={},
        ) -> Dict[str, np.ndarray]:
    rotation_transformer = RotationTransformer(
        from_rep='axis_angle', to_rep='rotation_6d')
    segmentation_labels = task_conf.get('segmentation_labels', None)
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    device = 'cuda'
    if 'robot_eef_pose' in env_obs.keys() and 'keypoint' in env_obs.keys():
        obs_dict_np = {'obs': env_obs['keypoint_in_world']}
        return obs_dict_np
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
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
        elif type == 'low_dim':
            if 'tracks' in key or 'vis' in key:
                if 'vis' in key:
                    continue
                camera = '_'.join(key.split('_')[:2])
                camera_img = env_obs[f'{camera}_color']
                obs_len, H, W, C = camera_img.shape
                num_points = 32
                hand_masks, object_masks = zip(*[get_segmentation_mask(img[..., ::-1], segmentation_labels, 'tmp', save_img=False) for img in camera_img])
                hand_points = np.concatenate(
                    [
                        np.expand_dims(
                            sample_from_mask(
                                np.expand_dims(hand_mask, axis=-1) * 255,
                                num_samples=int(obs_shape_meta[key].shape[0] / 2)
                            ), axis=0
                        ).astype(np.float32)
                        for hand_mask in hand_masks
                    ], axis=0
                )
                object_points = np.concatenate(
                    [
                        np.expand_dims(
                            sample_from_mask(
                                np.expand_dims(object_mask, axis=-1) * 255,
                                num_samples=int(obs_shape_meta[key].shape[0] / 2)
                            ), axis=0
                        ).astype(np.float32)
                        for object_mask in object_masks
                    ], axis=0
                )
                all_points =(np.concatenate([hand_points, object_points], axis=1))

                all_points[..., 0] /= H
                all_points[..., 1] /= W
                

                all_vis = np.ones(all_points.shape[:-1])
                
                # tracks, vis = sample_tracks_fps(all_points, all_vis, num_samples=num_points)
                obs_dict_np[f'{camera}_tracks']= all_points
                obs_dict_np[f'{camera}_vis'] = all_vis
                continue

            this_data_in = env_obs[key]
            if 'pose' in key and shape == (2,):
                # take X,Y coordinates
                this_data_in = this_data_in[...,[0,1]]
            if key =='robot_eef_pose':
                this_data_in = _convert_actions(this_data_in, rotation_transformer, 'cartesian_action')
                this_data_in = this_data_in[:, :shape_meta['obs']['robot_eef_pose'].shape[0]]
            obs_dict_np[key] = this_data_in
        elif type == 'feature' or type == 'sparse_feature':
            dataset_path = task_conf.dataset_path

            mast3r_size = max(shape_meta['obs']['camera_features']['original_shape'])
            mast3r_batch = shape_meta['obs']['camera_features']['info']['batch_size']

            # reference_img_paths = shape_meta['obs']['camera_features']['info']['reference_img_names']
            # reference_path = [os.path.join(dataset_path, img_name) for img_name in  reference_img_paths]
            # reference_imgs = load_images_mast3r(reference_path, size=mast3r_size, verbose=False, device=device)
            feature_view_keys = shape_meta['obs']['camera_features']['info']['view_keys']
            
            test_feature_view_key = real_obs_info['feature_view_key'] # feature_view_keys[1] 
            segment = shape_meta['obs']['camera_features']['info'].get("segment", False)

            
            view_image = env_obs[f'{test_feature_view_key}_color'] # (T, H ,W, C)
            
            
            # # original mask for both training and evaluating
            # remove_labels = ['human.hands.arms.sleeve.shirt.pants.hoody.jacket.robot.kinova.ur5e.finger pads']
            
            # # mask for training
            # remove_labels = ['diagonal human arm']
            
            # # mask label for evaluating
            remove_labels = ['a vertical cylinder']
            
            
            if segment:
                color_seq = np.array([
                    get_segmentation_mask(
                        frame,  # Each image is already in BGR format
                        segmentation_labels=remove_labels,
                        saved_path=None,
                        grounding_dino_model=real_obs_info['grounding_dino_model'],
                        sam_predictor=real_obs_info['sam_predictor'],
                        debug=False,
                        save_img=False,
                        remove_segmented=True
                    )[1][..., ::-1]
                    for frame in view_image[..., ::-1]
                ])
            else:
                color_seq = view_image
            # cv2.imshow('color_seq', color_seq[0])

            mast3r_imgs = load_images_mast3r(color_seq, size=mast3r_size, verbose=False)
            c,h,w = tuple(shape_meta['obs']['camera_features']['shape'])

            if shape_meta['obs']['camera_features']['info']['raw_image']:
                obs_dict_np[key] = F.interpolate(torch.concat([ImgNorm_inv(mast3r_imgs[i]['img']) for i in range(len(mast3r_imgs))]),size=(h,w), mode='area').cpu().numpy()

            else:
                mast3r_model = real_obs_info['mast3r_model']

                ref_img = None # reference_imgs[0]

                pca = real_obs_info['pca']
                                
                rm_irrel = shape_meta['obs']['camera_features']['info'].get('rm_irrel', False)

                mast3r_pairs = [tuple([ref_img] + [single_img]) for single_img in mast3r_imgs]

                mast3r_output = [inference([mast3r_pair], mast3r_model, device, batch_size=mast3r_batch, verbose=False) for mast3r_pair in mast3r_pairs]
                if type == 'sparse_feature':
                    # features = [feature_extractor.get_feature_representation(ImgNorm_inv(img_data['img']), (mast3r_size,mast3r_size)) for img_data in mast3r_imgs]
                    sparse_feature_num = shape_meta['obs']['camera_features']['sparse_shape'][0]
                    descriptor = [generate_correspondence(
                                    output['pred1']['desc'][0], 
                                    output['pred2']['desc'][0], 
                                    output['view1']['img'], 
                                    output['view2']['img'], 
                                    device='cuda', n_viz=sparse_feature_num,
                                    target_size=(h,w),
                                    rm_irrel=rm_irrel)[1][1] for i,output in enumerate(mast3r_output)]
                    obs_dict_np[key] = np.concatenate(descriptor).transpose(0, 3, 1, 2).astype(np.float32)

                else:
                    mast3r_pred2 = torch.stack([output['pred2']['desc'] for output in mast3r_output])
                    descriptor = mast3r_pred2.squeeze(0).detach().numpy().astype(np.float32)

                    reduced_descriptors = pca.transform(descriptor)
                    reduced_descriptors = reduced_descriptors.compute()
                    obs_dict_np[key] = reduced_descriptors.squeeze().transpose(0,3,1,2)
    return obs_dict_np

def flow_to_rigid_transform(track_model, env_obs, track_conf, n_action_steps):
    cam_name = track_conf.dataset_cfg.views[0]

    num_points = 32
    fs = 16
    segmentation_labels = ['a nail hammer']
    
    ho, wo = 128, 128
    obs_dict_np = dict()
    
    for img_key in [f'{cam_name}_color', f'{cam_name}_depth']:
        this_imgs_in = env_obs[img_key]
        if 'color' in img_key:
            t,hi,wi,ci = this_imgs_in.shape
        obs_dict_np[img_key] = np.concatenate([np.expand_dims(cv2.resize(img, (ho, wo), interpolation=cv2.INTER_LANCZOS4) , axis=0 )for img in this_imgs_in])
    
    original_intrinsics = env_obs[f'{cam_name}_intrinsics'][0]

    updated_intrinsics = update_intrinsics(original_intrinsics, (wi, hi), (wo, ho))
    obs_dict_np[f'{cam_name}_intrinsics'] = np.repeat(updated_intrinsics[np.newaxis, ...], t, axis=0)
    obs_dict_np[f'{cam_name}_extrinsics'] = env_obs[f'{cam_name}_extrinsics']

    tool_masks = np.array([get_segmentation_mask(cv2.resize(img, (wo, ho), interpolation=cv2.INTER_LANCZOS4)[..., ::-1], segmentation_labels, 'tmp', save_img=False)[0] for img in obs_dict_np[f'{cam_name}_color'] ])
    depth_masks = obs_dict_np[f'{cam_name}_depth'] > 0  # Shape: (2, 128, 128)
    tool_masks = (depth_masks & tool_masks).astype(np.uint8)  # Shape: (2, 128, 128)

    tool_points = np.concatenate(
        [
            np.expand_dims(
                sample_from_mask(
                    np.expand_dims(tool_mask, axis=-1) * 255,
                    num_samples=num_points
                ), axis=0
            ).astype(np.float32)
            for tool_mask in tool_masks
        ], axis=0
    )
    tool_points[..., 0] /= wo
    tool_points[..., 1] /= ho
    tool_points = torch.from_numpy(tool_points).float().cuda()
    
    track_rgb = torch.from_numpy(obs_dict_np[f'{cam_name}_color']).float().cuda()
    track_rgb = track_rgb.unsqueeze(1).expand(-1, fs, -1, -1, -1)       
    track_rgb = rearrange(track_rgb,'To fs h w c -> To fs c h w')
    track_to_pred = tool_points.unsqueeze(1).expand(-1, fs, -1, -1)
    
    pred_tr, _ = track_model.reconstruct(track_rgb, track_to_pred, p_img=0) # (b v t) tl n d
    pred_tr[..., 0], pred_tr[..., 1] = pred_tr[..., 0] * wo, pred_tr[..., 1] * ho
    pred_tr = pred_tr.long()
    pred_tr[..., 0], pred_tr[..., 1] = torch.clamp(pred_tr[..., 0], 0, wo-1), torch.clamp(pred_tr[..., 1], 0, ho-1)
    
    points_2d_init = pred_tr[-1, 0].clone().cpu().numpy()

    points_2d_after = pred_tr[-1, n_action_steps-1].clone().cpu().numpy()
        
    
    intrinsics = obs_dict_np[f'{cam_name}_intrinsics'][-1]    
    points_3d_init = convert_2d_to_3d(points_2d_init, obs_dict_np[f'{cam_name}_depth'][-1], intrinsics)
    
    # T P3_0 = P3_t
    # K T P3_0 = (K P3_t) = P2_t
    retval, rvec, tvec = cv2.solvePnP(points_3d_init, points_2d_after.astype(np.float32), intrinsics, distCoeffs=np.array([0,0,0,0,0]))
    rotmat, _ = cv2.Rodrigues(rvec)
    rigid_transform = np.hstack((rotmat, tvec/1000))
    
    return rigid_transform

def flow_to_9d_transform(pred_tr, env_obs, n_action_steps, n_obs_steps):
    wi, hi = 640, 480
    pattern = re.compile(r'camera_\d+_color')
    rgb_key = [key for key in env_obs if pattern.match(key)][0]
    depth_key = rgb_key.replace('color', 'depth')
    intrinsics_key = rgb_key.replace('color', 'intrinsics')

    batch_size, horizon, channel, ho, wo = env_obs[rgb_key].shape
    _, _, _,fs, num_points, _ = pred_tr.shape
    # TODO: improved the speed
    pred_tr[..., 0], pred_tr[..., 1] = pred_tr[..., 0] * wo, pred_tr[..., 1] * ho
    pred_tr = pred_tr.long()
    pred_tr[..., 0], pred_tr[..., 1] = torch.clamp(pred_tr[..., 0], 0, wo-1), torch.clamp(pred_tr[..., 1], 0, ho-1)
    all_tracks_transform_9d = []
    for obs_idx in range(n_obs_steps):
        points_2d_init = pred_tr[:, 0, obs_idx, 0].clone().detach()  # Points at the first step
        points_2d_after = pred_tr[:, 0, obs_idx, n_action_steps-1].clone().detach()  # Points after n_action_steps
    
        original_intrinsics = env_obs[intrinsics_key][:,obs_idx]
        updated_intrinsics = update_intrinsics(original_intrinsics, (wi, hi), (wo, ho))

        depth_map = env_obs[depth_key].squeeze(2)[:,obs_idx]
        # valid_mask = depth_map > 0  # Shape: (batch_size, 128, 128)
        # for b in range(batch_size):
        #     valid_pt_mask = valid_mask[b][points_2d_init[b][:,1], points_2d_init[b][:,0]]        
        
        points_3d_init = convert_2d_to_3d(points_2d_init, depth_map, updated_intrinsics)

        # Create a mask to filter out points with z = 0
        valid_mask = points_3d_init[..., 2] > 0

        padding_value=1
        # Prepare padded points containers
        padded_points_3d_init = torch.full((batch_size, num_points, 3), padding_value, device=points_3d_init.device)
        padded_points_2d_after = torch.full((batch_size, num_points, 2), padding_value, device=points_2d_after.device)
        padded_weights = torch.zeros(batch_size, num_points, device=points_3d_init.device)  # Set all weights to 0 initially

        for i in range(batch_size):
            valid_indices = valid_mask[i]
            num_valid_points = valid_indices.sum()

            # Extract valid points for this batch
            valid_points_3d_init = points_3d_init[i][valid_indices]
            valid_points_2d_after = points_2d_after[i][valid_indices]

            # Pad the valid points into the padded containers
            padded_points_3d_init[i, :num_valid_points] = valid_points_3d_init
            padded_points_2d_after[i, :num_valid_points] = valid_points_2d_after
            
            padded_weights[i, :num_valid_points] = 1.0

        # T P3_0 = P3_t
        # K T P3_0 = (K P3_t) = P2_t
        projection = efficient_pnp(
            padded_points_3d_init.float(), 
            padded_points_2d_after.float(), 
            weights=padded_weights.float()
        )
        proj_rot = projection.R          # Rotation matrices (minibatch, 3, 3)
        proj_trans = projection.T          # Translation vectors (minibatch, 3)
        proj_matrix = torch.cat([proj_rot, proj_trans.unsqueeze(-1)], dim=-1)  # (batch_size, 3, 4)
        tracks_transform = torch.bmm(torch.inverse(updated_intrinsics)  , proj_matrix)  # (batch_size, 3, 4)

        tracks_rot = tracks_transform[:, :, :3]  # Shape: (batch_size, 3, 3)
        tracks_trans = tracks_transform[:, :, 3]    # Shape: (batch_size, 3)

        # Step 2: Convert the rotation matrix to 6D rotation representation
        R_6d = matrix_to_rotation_6d(tracks_rot)  # Shape: (batch_size, 6)

        # Step 3: Concatenate the position (T) and the 6D rotation (R_6d)
        tracks_transform_9d = torch.cat([tracks_trans, R_6d], dim=-1)  # Shape: (batch_size, 9)
        all_tracks_transform_9d.append(tracks_transform_9d)
    all_tracks_transform_9d = torch.stack(all_tracks_transform_9d, dim=1)  # Shape: (batch_size, n_obs_steps, 9)
    return all_tracks_transform_9d


def get_tool_in_cam(pose_model, segmentation_labels, obs, shape_meta, to_origin, bbox, camera_key, vis_img=True, init=False, grounding_dino_model=None, sam_predictor=None):
    # pattern = re.compile(r'camera_\d')
    # if 'camera_features' in shape_meta['obs']:
    #     camera_key = [key for key in shape_meta['obs']['camera_features']['info']['view_keys'] if pattern.match(key)][0]
    # else:
    #     camera_color_key = [key for key in shape_meta['obs'].keys() if pattern.match(key)][0]
    #     camera_key = camera_color_key.split('_')[0] + '_' + camera_color_key.split('_')[1]
    rgb = obs[camera_key + '_color'][-1].copy()
    depth = obs[camera_key + '_depth'][-1].astype(np.float64)/1000
    intrinsics = obs[camera_key + '_intrinsics'][-1].astype(np.float64)
    if init==True:
        tool_mask, _ = get_segmentation_mask(rgb[..., ::-1], segmentation_labels, 'tmp.png', grounding_dino_model, sam_predictor)
        pose = pose_model.register(K=intrinsics, rgb=rgb, depth=depth, ob_mask=tool_mask, iteration=5)
    else:
        pose = pose_model.track_one(rgb=rgb, depth=depth, K=intrinsics, iteration=2)
    if vis_img:
        center_pose = pose@np.linalg.inv(to_origin)
        vis_rgb = draw_posed_3d_box(intrinsics, img=rgb, ob_in_cam=center_pose, bbox=bbox)
        vis_rgb = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=intrinsics, thickness=3, transparency=0, is_input_rgb=True)

    return pose, vis_rgb

def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res
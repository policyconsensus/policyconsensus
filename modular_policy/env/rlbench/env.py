import torch
import numpy as np
import gymnasium
import hydra
import gc
import os
import joblib
from rlbench.observation_config import ObservationConfig
from rlbench.backend.observation import Observation
from rlbench.action_modes.action_mode import ActionMode
from rlbench.environment import Environment
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.objects.dummy import Dummy
from pyrep.const import RenderMode

from modular_policy.common.lang_emb import get_lang_emb
from modular_policy.common.pointcloud_sampler import pointcloud_subsampling
from modular_policy.common.dinov2_util import (
    load_dino_model, compute_dino_features, pca_dim_reduction)
from modular_policy.env.rlbench.action_modes import JointPositionActionMode

from typing import Optional, List


# NOTE: tabletop at z=0.75
TASK_BOUNDS = {
    'default': [
        [-0.30, -0.60, 0.76],    # lb
        [ 0.80,  0.60, 1.75]     # ub
    ],
}


class RlbenchEnv(gymnasium.Env):
    metadata = {
        "render_modes": ['rgb_array', 'depth_array'],
        "render_fps": 10
    }

    def __init__(self,
        task_name: Optional[str] = None,
        # NOTE: image_size is always 128x128
        image_size: int = 128,
        num_points: Optional[int] = None,
        enable_depth: bool = False,
        enable_dino: bool = False,
        process_dino: bool = False,
        pca_reduction: Optional[int] = None,
        seed: Optional[int] = None,
        camera_names: List[str] = ['left_shoulder','right_shoulder','overhead','wrist','front'],
        robot_state_ports: List[str] = [
            'joint_positions', 'joint_velocities', 'joint_forces',
            'gripper_open', 'gripper_pose', 'gripper_joint_positions',
            'gripper_touch_forces'
        ],
        video_resolution: int = 512,
        max_episode_steps: int = 250,
        zarr_path: Optional[str] = None,  # for pca loading
        action_mode: Optional[ActionMode] = None,
        image_scale: float = 255.0,
    ):
        assert enable_depth or num_points is None, \
            "num_points must be None if enable_depth is False"
        assert not (enable_dino and not enable_depth), \
            "must enable_depth to use DINO features"
        super().__init__()

        if action_mode is None:
            action_mode = JointPositionActionMode()
        
        # config DINO model
        dino_model = None
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if enable_dino:
            dino_model = load_dino_model('dinov2_vits14')
            dino_model = dino_model.to(device)

        self.dino_model = dino_model
        self.device = device
        self.action_mode = action_mode
        self.camera_names = camera_names
        self.robot_state_ports = robot_state_ports
        self.num_points = num_points
        self.pca_reduction = pca_reduction
        self.image_size = image_size
        self.enable_depth = enable_depth
        self.enable_dino = enable_dino
        self.process_dino = process_dino

        self.video_resolution = video_resolution
        self.max_episode_steps = max_episode_steps
        self.image_scale = image_scale
        self.done = False

        self.pca = None
        if self.enable_dino and zarr_path is not None:
            pca_path = os.path.join(zarr_path, 'pca_dino_3d.pkl')
            if os.path.exists(pca_path):
                print(f"Loading PCA from {pca_path}")
                self.pca = joblib.load(pca_path)
            else:
                print(f"[Warning] PCA file not found at {pca_path}. DINO features will not be reduced.")

        # observation config
        obs_config = ObservationConfig()
        for name in camera_names:
            obs_config.__dict__[f"{name}_camera"].rgb = True
            obs_config.__dict__[f"{name}_camera"].depth = enable_depth
            obs_config.__dict__[f"{name}_camera"].point_cloud = num_points is not None
            obs_config.__dict__[f"{name}_camera"].image_size = (image_size, image_size)
        for name in robot_state_ports:
            obs_config.__dict__[name] = True

        # coppelia engine setup
        self.rlbench_env = Environment(
            action_mode=action_mode,
            obs_config=obs_config,
            headless=True
        )
        self.rlbench_env.launch()

        # task setup
        if task_name is not None:
            self.set_task(task_name, seed=seed)

    
    def _reset_task_vars(self):
        self.task_name = None
        # self.task_name_emb = None
        self.task_env = None
        self.task_bbox = None
        self.recording_camera = None
        self.observation_space = None
        self.action_space = None
        self.cur_step = 0
        self.done = False
        gc.collect()

    
    def set_task(self, task_name: str, seed: Optional[int] = None):
        # clean up if any
        self._reset_task_vars()

        self.task_name = task_name
        # self.task_name_emb = get_lang_emb(task_name).cpu().numpy()
        self.task_env = self.rlbench_env.get_task(
            hydra.utils.get_class(
                f"rlbench.tasks.{task_name}."
                f"{''.join([word.capitalize() for word in task_name.split('_')])}"
            )
        )
        self.task_bbox = np.array(TASK_BOUNDS.get(task_name, TASK_BOUNDS['default']))
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)

        # video recording camera setup
        dummy_placeholder = Dummy("cam_cinematic_placeholder")
        self.recording_camera = VisionSensor.create(
            [self.video_resolution, self.video_resolution])
        self.recording_camera.set_pose(dummy_placeholder.get_pose())
        self.recording_camera.set_render_mode(RenderMode.OPENGL3)

        # setup gym spaces 
        _, obs = self.task_env.reset()
        sample_obs_dict = self._extact_obs(obs)
        self.observation_space = gymnasium.spaces.Dict({
            key: gymnasium.spaces.Box(
                low=0, high=255, 
                shape=value.shape, dtype=value.dtype
            ) if 'rgb' in key else gymnasium.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=value.shape, dtype=value.dtype
            ) for key, value in sample_obs_dict.items()
        })
        self.action_space = gymnasium.spaces.Box(
            low=np.float32(-2*np.pi), high=np.float32(2*np.pi),
            shape=self.rlbench_env.action_shape, dtype=np.float32
        )


    def _extact_obs(self, rlbench_obs: Observation):
        obs_dict = {}

        # robot state
        for port_name in self.robot_state_ports:
            state_data = getattr(rlbench_obs, port_name, None)
            if state_data is not None:
                state_data = np.float32(state_data)
                if np.isscalar(state_data):
                    state_data = np.asarray([state_data])
                obs_dict[port_name] = state_data

        # language embedding
        # obs_dict["language"] = self.task_name_emb

        # rgb, depth
        for name in self.camera_names:
            rgb = getattr(rlbench_obs, f"{name}_rgb").transpose(2, 0, 1).astype(np.float32)
            # Apply image scaling
            if self.image_scale != 255.0:
                # Scale from [0, 255] to desired scale
                rgb = rgb * (self.image_scale / 255.0)
            obs_dict[f"{name}_rgb"] = rgb
            if self.enable_depth:
                obs_dict[f"{name}_depth"] = np.expand_dims(getattr(rlbench_obs, f"{name}_depth"), axis=0)

        # fused pointcloud
        if self.num_points is not None:
            pointcloud = np.concatenate([
                getattr(rlbench_obs, f"{name}_point_cloud").reshape(-1, 3)
                for name in self.camera_names
            ])
            assert len(pointcloud) == len(self.camera_names) * self.image_size**2
            
            if self.enable_dino:
                # compute DINO features
                feats, _ = compute_dino_features(self.dino_model,   # (N, D, H, W)
                    images=torch.from_numpy(
                        np.stack([
                            obs_dict[f'{cam_name}_rgb'] 
                            for cam_name in self.camera_names
                        ], axis=0) / self.image_scale
                    ).to(dtype=torch.float32, device=self.device)
                )
                feats = feats.permute(0, 2, 3, 1).view(-1, self.dino_model.num_features).cpu().numpy()

                pointcloud = np.concatenate([pointcloud, feats], axis=-1)

            # crop and subsample
            pointcloud = pointcloud[np.all(
                (pointcloud[:, :3] >= self.task_bbox[0]) &
                (pointcloud[:, :3] <= self.task_bbox[1]),
            axis=-1)]
            
            if pointcloud.shape[0] < self.num_points:
                indices = np.random.choice(pointcloud.shape[0], size=self.num_points, replace=True)
                pointcloud = pointcloud[indices]
            else:
                pointcloud = pointcloud_subsampling(pointcloud, self.num_points, method='fps')
            
            if self.enable_dino and self.pca is not None:
                # PCA reduction if applicable
                feats = pointcloud[:, 3:]
                feats = self.pca.transform(feats)
                pointcloud = np.concatenate([pointcloud[:, :3], feats], axis=-1)

            obs_dict["fused_pointcloud"] = pointcloud[:, :3]
            if self.enable_dino:
                obs_dict["fused_dino"] = pointcloud

        return obs_dict
    

    def render(self, mode='rgb_array'):
        # NOTE: only for video recording wrapper
        assert mode == 'rgb_array'
        frame = self.recording_camera.capture_rgb()
        frame = np.clip((frame * 255.).astype(np.uint8), 0, 255)
        return frame
    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        description, obs = self.task_env.reset()
        self.cur_step = 0
        self.done = False
        return self._extact_obs(obs), {'text_description': description}
    
    
    def step(self, action: np.ndarray):
        obs, reward, terminated = self.task_env.step(action)
        self.cur_step += 1
        self.done = self.done or terminated or (reward >= 1) \
            or (self.cur_step >= self.max_episode_steps)
        return self._extact_obs(obs), reward, self.done, False, {}


    def close(self) -> None:
        self.rlbench_env.shutdown()

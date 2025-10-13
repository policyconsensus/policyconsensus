import torch
import torch.nn as nn
import numpy as np
from pytorch3d.ops import sample_farthest_points

from modular_policy.common.pytorch_util import replace_submodules
from modular_policy.model.perception.sensory_encoder import BaseSensoryEncoder
try:
    from modular_policy.model.perception.graph_util import build_graph, build_graph_encoder
except ImportError:
    build_graph = None
    build_graph_encoder = None
try:
    from modular_policy.model.perception.point_transformer.model import PointTransformerV3
except ImportError:
    PointTransformerV3 = None

from typing import List, Optional, Union, Dict


class DinoPointcloudEncoder(BaseSensoryEncoder):
    def __init__(self,
        shape_meta: Dict,
        out_channels: int,
    ):
        super().__init__()

        # parse shape / obs config
        pointcloud_keys = list()
        key_shape_map = dict()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr['type']
            if type == 'dino_pointcloud':
                pointcloud_keys.append(key)
                key_shape_map[key] = shape

        # # currently, fuse -> normalize -> encode
        # if len(pointcloud_keys) != 1:
        #     raise RuntimeError(
        #         f"There should be exactly 1 (fused) pointcloud observation, "
        #         f"but found: {pointcloud_keys}."
        #     )
        
        # attr assignment
        self.pointcloud_keys = pointcloud_keys
        self.key_shape_map = key_shape_map
        self.out_channels = out_channels


    def encode(self, points: torch.Tensor) -> torch.Tensor:
        # input: (B, N, d), output: (B, out_channels)
        raise NotImplementedError


    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # output: (B, feature_dim)
        batch_size = None
        features = list()
        for key in self.pointcloud_keys:
            points = obs_dict[key]
            if batch_size is None:
                batch_size = points.size(0)
            else:
                assert batch_size == points.size(0)
            assert points.shape[1:] == self.key_shape_map[key]
            features.append(self.encode(points))
        return dict(zip(self.pointcloud_keys, features))
    

    def output_feature_dim(self):
        return {k: self.out_channels for k in self.pointcloud_keys}
    

    def modalities(self):
        return ['dino_pointcloud',]


class DinoDp3PointcloudEncoder(DinoPointcloudEncoder):
    def __init__(
        self,
        shape_meta: Dict,
        out_channels: int,
        # backbone arguments
        layer_size: List[int] = [64, 128, 256],
        use_layer_norm: bool = True,
    ):
        assert len(layer_size) == 3, "len(layer_size) must be 3"
        super().__init__(shape_meta, out_channels)

        # DP3 backbone specification
        self.encoder = nn.Sequential(
            nn.Linear(self.key_shape_map[self.pointcloud_keys[0]][-1], layer_size[0]),
            nn.LayerNorm(layer_size[0]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(layer_size[0], layer_size[1]),
            nn.LayerNorm(layer_size[1]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(layer_size[1], layer_size[2]),
            nn.LayerNorm(layer_size[2]) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
        )
        self.projector = nn.Sequential(
            nn.Linear(layer_size[2], out_channels),
            nn.LayerNorm(out_channels) if use_layer_norm else nn.Identity()
        )

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        B = points.size(0)
        return self.projector(
            torch.max(
                self.encoder(points),
                dim=1
            ).values
        ).view(B, self.out_channels)



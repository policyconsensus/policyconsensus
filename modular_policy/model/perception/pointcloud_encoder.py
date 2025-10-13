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



def fuse_pointcloud(
    pointclouds: List[Union[np.ndarray, torch.Tensor]], 
    num_points: Optional[int]
) -> Union[np.ndarray, torch.Tensor]:
    # input: List[(*, N, d)]
    # output: (*, num_points, d) or (*, sum(N), d)
    is_np = False
    is_cpu = True
    if isinstance(pointclouds[0], np.ndarray):
        result_pcd = np.concatenate(pointclouds, axis=-2)
        is_np = True
    elif isinstance(pointclouds[0], torch.Tensor):
        result_pcd = torch.cat(pointclouds, dim=-2)
        is_cpu = not result_pcd.is_cuda
    else:
        raise ValueError("Unsupported type")
    
    unsqueezed = False
    if result_pcd.ndim == 2:
        if is_np:
            result_pcd = result_pcd[np.newaxis]
        else:
            result_pcd = result_pcd.unsqueeze(0)
        unsqueezed = True
    
    if num_points is not None:
        if is_np:
            result_pcd = torch.from_numpy(result_pcd)
        if torch.cuda.is_available():
            result_pcd = result_pcd.cuda()
        result_pcd = sample_farthest_points(
            result_pcd, K=num_points, 
            random_start_point=True
        )
        if is_cpu:
            result_pcd = result_pcd.cpu()
        if is_np:
            result_pcd = result_pcd.numpy()

    if unsqueezed:
        result_pcd = result_pcd.squeeze(0)

    return result_pcd


class PointcloudEncoder(BaseSensoryEncoder):
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
            if type == 'pointcloud':
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
        return ['pointcloud',]


class Dp3PointcloudEncoder(PointcloudEncoder):
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


class PointTransformerEncoder(PointcloudEncoder):
    def __init__(self,
        shape_meta: Dict,
        out_channels: int,
        # backbone arguments
        grid_size: float = 0.01,
        use_group_norm: bool = True,
        use_layer_norm: bool = True,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2),
        enc_depths=(2, 6, 2),
        enc_channels=(32, 64, 128),
        enc_num_head=(2, 4, 8),
        enc_patch_size=(1024, 1024, 1024),
        dec_depths=(2, 2),
        dec_channels=(64, 64),
        dec_num_head=(4, 4),
        dec_patch_size=(1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ):
        if PointTransformerV3 is None:
            raise ImportError(
                "PointTransformerV3 is not available. "
                "Refer to https://github.com/Pointcept/PointTransformerV3 for installation."
            )

        super().__init__(shape_meta, out_channels)
        
        # Point Transformer V3 backbone specification
        self.pointtransformerv3 = PointTransformerV3(
            in_channels=self.key_shape_map[self.pointcloud_keys[0]][-1],
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            dec_depths=dec_depths,
            dec_channels=dec_channels,
            dec_num_head=dec_num_head,
            dec_patch_size=dec_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cls_mode=False,  # to enable point transformer v3's decoder
            pdnorm_bn=pdnorm_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=pdnorm_decouple,
            pdnorm_adaptive=pdnorm_adaptive,
            pdnorm_affine=pdnorm_affine,
            pdnorm_conditions=pdnorm_conditions,
        )
        self.projector = nn.Sequential(
            nn.Linear(dec_channels[0], out_channels),
            nn.LayerNorm(out_channels) if use_layer_norm else nn.Identity()
        )

        # replace BatchNorm with GroupNorm if any
        if use_group_norm:
            self.pointtransformerv3 = replace_submodules(
                root_module=self.pointtransformerv3,
                predicate=lambda x: isinstance(x, nn.BatchNorm1d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16,
                    num_channels=x.num_features,
                )
            )

    
    def encode(self, points: torch.Tensor) -> torch.Tensor:
        B, N, D = points.size()
        points = points.view(B * N, D)
        offset = torch.arange(B) * N + N
        return self.projector(
            torch.max(
                self.pointtransformerv3({
                    "feat": points,
                    "coord": points[:, :3],
                    "grid_size": self.grid_size,
                    "offset": offset,
                }).feat.view(B, N, -1),
                dim=1
            ).values
        ).view(B, self.out_channels)


class ParticleGraphEncoder(PointcloudEncoder):
    def __init__(self,
        shape_meta: Dict,
        out_channels: int,
        # backbone arguments
        knn: int = 5,
        num_propagation_steps: int = 3,
    ):
        if build_graph is None or build_graph_encoder is None:
            raise ImportError(
                "ParticleGraphEncoder is not available. "
                "PyG installation is required."
            )

        super().__init__(shape_meta, out_channels)

        node_input_dim = self.key_shape_map[self.pointcloud_keys[0]][-1] - 3
        assert node_input_dim >= 0, "Point feature dimension must be non-negative"
        edge_input_dim = 3
        if node_input_dim == 0:
            node_input_dim = 1  # dummy feature
        else:
            edge_input_dim += node_input_dim * 2

        self.graph_encoder = build_graph_encoder(
            node_input_dim=node_input_dim,
            node_output_dim=out_channels,
            edge_input_dim=edge_input_dim,
            num_propagation_steps=num_propagation_steps,
        )
        self.decoder = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        
        
    def encode(self, points: torch.Tensor) -> torch.Tensor:
        B, N, D = points.size()
        if D > 3:
            point_feats = points[..., 3:]
        else:
            point_feats = None
        nodes, edges, edge_indices = build_graph(
            point_coords=points[..., :3],
            point_feats=point_feats,
            knn=self.knn
        )
        return self.decoder(
            torch.max(
                self.graph_encoder(nodes, edges, edge_indices).view(B, N, -1),
                dim=1
            ).values
        ).view(B, self.out_channels)

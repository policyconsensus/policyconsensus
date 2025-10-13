from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_nets as rmon
from modular_policy.common.pytorch_util import replace_submodules
from modular_policy.model.perception.sensory_encoder import BaseSensoryEncoder
from modular_policy.model.perception.crop_randomizer import CropRandomizer

from typing import Dict, Tuple, Union


class DepthEncoder(BaseSensoryEncoder):
    def __init__(self):
        super().__init__()

    def modalities(self):
        return ['depth',]


class RobomimicDepthEncoder(DepthEncoder):
    def __init__(self,
        shape_meta: dict,
        crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        use_group_norm: bool=True,
        eval_fixed_crop: bool = False,
        share_depth_model: bool=False,
    ):
        super().__init__()

        depth_ports = list()
        port_shape = dict()
        for key, attr in shape_meta['obs'].items():
            type = attr['type']
            shape = attr['shape']
            if type == 'depth':
                depth_ports.append(key)
                port_shape[key] = shape

        # init global state
        ObsUtils.initialize_obs_modality_mapping_from_dict({"depth": depth_ports})

        def crop_randomizer(shape, crop_shape):
            if crop_shape is None:
                return None
            return rmbn.CropRandomizer(
                input_shape=shape,
                crop_height=crop_shape[0],
                crop_width=crop_shape[1],
                num_crops=1,
                pos_enc=False,
            )
            
        def visual_net(shape, crop_shape):
            if crop_shape is not None:
                shape = (shape[0], crop_shape[0], crop_shape[1])
            net = rmbn.VisualCore(
                input_shape=shape,
                feature_dimension=64,
                backbone_class='ResNet18Conv',
                backbone_kwargs={
                    'input_channels': shape[0],
                    'input_coord_conv': False,
                },
                pool_class='SpatialSoftmax',
                pool_kwargs={
                    'num_kp': 32,
                    'temperature': 1.0,
                    'noise': 0.0,
                },
                flatten=True,
            )
            return net

        obs_encoder = rmon.ObservationEncoder()
        if share_depth_model:
            this_shape = port_shape[depth_ports[0]]
            net = visual_net(this_shape, crop_shape)
            obs_encoder.register_obs_key(
                name=depth_ports[0],
                shape=this_shape,
                net=net,
                randomizer=crop_randomizer(this_shape, crop_shape),
            )
            for port in depth_ports[1:]:
                assert port_shape[port] == this_shape
                obs_encoder.register_obs_key(
                    name=port,
                    shape=this_shape,
                    randomizer=crop_randomizer(this_shape, crop_shape),
                    share_net_from=depth_ports[0],
                )
        else:
            for port in depth_ports:
                shape = port_shape[port]
                net = visual_net(shape, crop_shape)
                obs_encoder.register_obs_key(
                    name=port,
                    shape=shape,
                    net=net,
                    randomizer=crop_randomizer(shape, crop_shape),
                )

        if use_group_norm:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16,
                    num_channels=x.num_features,
                )
            )

        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_encoder.make()
        self.encoder = obs_encoder
        self.depth_keys = obs_encoder.obs_shapes.keys()

    def forward(self, obs_dict) -> Dict[str, torch.Tensor]:
        output = self.encoder(obs_dict) # (B,N*D)
        B = output.shape[0]
        N = len(self.depth_keys)
        output = output.reshape(B,N,-1)
        return dict(zip(self.depth_keys, output.unbind(1)))
    
    
    @torch.no_grad()
    def output_feature_dim(self):
        D = self.encoder.output_shape()[0]
        N = len(self.depth_keys)
        return dict(zip(self.depth_keys, [D // N] * N))

from typing import Dict, Tuple, Union
import torch
import torch.nn as nn
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_nets as rmon
from modular_policy.common.pytorch_util import replace_submodules
from modular_policy.model.perception.sensory_encoder import BaseSensoryEncoder


class TactileEncoder(BaseSensoryEncoder):
    def __init__(self):
        super().__init__()

    def modalities(self):
        return ['tactile',]


class RobomimicTactileEncoder(TactileEncoder):
    def __init__(self,
        shape_meta: dict,
        crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        use_group_norm: bool=True,
        eval_fixed_crop: bool = False,
        share_tactile_model: bool=False,
    ):
        super().__init__()

        # Filter for tactile modalities only - treat as RGB for processing
        tactile_ports = list()
        port_shape = dict()
        for key, attr in shape_meta['obs'].items():
            type = attr['type']
            shape = attr['shape']
            if type == 'rgb' and 'tactile' in key.lower():
                tactile_ports.append(key)
                port_shape[key] = shape

        # init global state - treat tactile as RGB for visual processing
        ObsUtils.initialize_obs_modality_mapping_from_dict({"rgb": tactile_ports})

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
                    'noise_std': 0.0,
                },
            )
            return net

        # Create observation encoder
        obs_encoder = rmon.ObservationEncoder()
        
        if share_tactile_model:
            for port in tactile_ports:
                if port == tactile_ports[0]:  # First tactile sensor
                    shape = port_shape[port]
                    net = visual_net(shape, crop_shape)
                    obs_encoder.register_obs_key(
                        name=port,
                        shape=shape,
                        net=net,
                        randomizer=crop_randomizer(shape, crop_shape),
                    )
                else:  # Share network from first tactile sensor
                    this_shape = port_shape[port]
                    obs_encoder.register_obs_key(
                        name=port,
                        shape=this_shape,
                        randomizer=crop_randomizer(this_shape, crop_shape),
                        share_net_from=tactile_ports[0],
                    )
        else:
            for port in tactile_ports:
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
                    eps=x.eps, 
                    affine=x.affine
                )
            )

        obs_encoder.make()
        self.obs_encoder = obs_encoder
        self.obs_keys = list(obs_encoder.obs_shapes.keys())
        self.tactile_keys = tactile_ports

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_dict = {k: obs[k] for k in self.obs_keys if k in obs}
        output = self.obs_encoder(obs_dict)  # (B,N*D)
        B = output.shape[0]
        N = len(self.tactile_keys)
        output = output.reshape(B, N, -1)
        return dict(zip(self.tactile_keys, output.unbind(1)))

    def output_feature_dim(self):
        D = self.obs_encoder.output_shape()[0]
        N = len(self.tactile_keys)
        return dict(zip(self.tactile_keys, [D // N] * N))
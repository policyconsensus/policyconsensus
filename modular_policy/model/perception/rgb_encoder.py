from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_nets as rmon
from modular_policy.common.pytorch_util import replace_submodules, dict_apply
from modular_policy.model.perception.sensory_encoder import BaseSensoryEncoder
from modular_policy.model.perception.crop_randomizer import CropRandomizer

from typing import Dict, Tuple, Union


def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet


def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model


class RgbEncoder(BaseSensoryEncoder):
    def __init__(self):
        super().__init__()

    def modalities(self):
        return ['rgb',]


class DpRgbEncoder(RgbEncoder):
    def __init__(self,
        shape_meta: dict,
        rgb_model: Union[nn.Module, Dict[str,nn.Module]],
        resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        random_crop: bool=True,
        # replace BatchNorm with GroupNorm
        use_group_norm: bool=False,
        # use single rgb model for all rgb inputs
        share_rgb_model: bool=False,
        # renormalize rgb input with imagenet normalization
        # assuming input in [0,1]
        imagenet_norm: bool=False
    ):
        """
        Assumes rgb input: B,C,H,W
        """
        super().__init__()

        rgb_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'unsupported')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
        rgb_keys = sorted(rgb_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.key_shape_map = key_shape_map


    def forward(self, obs_dict) -> Dict[str,torch.Tensor]:
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            features = feature.unbind(1)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        return dict(zip(self.rgb_keys, features))


    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        return dict_apply(example_output, lambda x: x.shape[1:])
    

    def output_feature_dim(self):
        output_shape = self.output_shape()
        return dict_apply(output_shape, lambda x: x[0])
    

class RobomimicRgbEncoder(RgbEncoder):
    def __init__(self,
        shape_meta: dict,
        crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
        use_group_norm: bool=True,
        eval_fixed_crop: bool = False,
        share_rgb_model: bool=False,
    ):
        super().__init__()

        rgb_ports = list()
        port_shape = dict()
        for key, attr in shape_meta['obs'].items():
            type = attr['type']
            shape = attr['shape']
            if type == 'rgb' and 'tactile' not in key.lower():
                rgb_ports.append(key)
                port_shape[key] = shape

        # init global state
        ObsUtils.initialize_obs_modality_mapping_from_dict({"rgb": rgb_ports})

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
        if share_rgb_model:
            this_shape = port_shape[rgb_ports[0]]
            net = visual_net(this_shape, crop_shape)
            obs_encoder.register_obs_key(
                name=rgb_ports[0],
                shape=this_shape,
                net=net,
                randomizer=crop_randomizer(this_shape, crop_shape),
            )
            for port in rgb_ports[1:]:
                assert port_shape[port] == this_shape
                obs_encoder.register_obs_key(
                    name=port,
                    shape=this_shape,
                    randomizer=crop_randomizer(this_shape, crop_shape),
                    share_net_from=rgb_ports[0],
                )
        else:
            for port in rgb_ports:
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
        self.rgb_keys = list(obs_encoder.obs_shapes.keys())


    def forward(self, obs_dict) -> Dict[str,torch.Tensor]:
        output = self.encoder(obs_dict) # (B,N*D)
        B = output.shape[0]
        N = len(self.rgb_keys)
        output = output.reshape(B,N,-1)
        return dict(zip(self.rgb_keys, output.unbind(1)))

    
    @torch.no_grad()
    def output_feature_dim(self):
        D = self.encoder.output_shape()[0]
        N = len(self.rgb_keys)
        return dict(zip(self.rgb_keys, [D // N] * N))

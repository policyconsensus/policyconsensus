from typing import Dict, List, Union, Optional, Tuple
import torch
import torch.nn as nn
import dill
import hydra
import omegaconf
from modular_policy.model.common.module_attr_mixin import ModuleAttrMixin
from modular_policy.model.common.normalizer import LinearNormalizer
from modular_policy.model.perception.sensory_encoder import BaseSensoryEncoder

class BasePolicy(ModuleAttrMixin):

    @classmethod
    def from_checkpoint(cls, checkpoint: str, output_dir: Optional[str] = None):
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        try:
            if cfg.training.use_ema:
                policy = workspace.ema_model
        except omegaconf.errors.ConfigAttributeError:
            pass
        return policy

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return:
            A dictionary containing the predicted action and optionally, modality importance data.
            {
                'action': torch.Tensor,
                'importance_data': {
                    'router_weights': Dict[str, float],
                    'attention_weights': torch.Tensor
                }
            }
        """
        raise NotImplementedError()

    def reset(self):
        pass

    def set_normalizer(self, normalizer: Union[LinearNormalizer, List[LinearNormalizer]]):
        raise NotImplementedError()

    def get_noise_prediction_network(self):
        raise NotImplementedError()
    
    def get_observation_encoder(self):
        raise NotImplementedError()
    
    def get_observation_modalities(self) -> List[str]:
        raise NotImplementedError()
    
    def get_observation_ports(self) -> List[str]:
        raise NotImplementedError()
    
    def get_policy_name(self) -> str:
        raise NotImplementedError()
    
    def create_dummy_observation(self,
        batch_size: int,
        horizon: int,
        obs_key_shapes: Dict[str, Tuple[int]],
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        obs_dict = dict()
        for obs_port, obs_shape in obs_key_shapes.items():
            obs_dict[obs_port] = torch.randn(
                size=(batch_size, horizon, *obs_shape),
            ).to(device)
        return obs_dict
    
    def create_observation_encoder(self,
        obs_encoders: List[BaseSensoryEncoder],
        modalities: List[str],
        obs_config: Optional[Dict[str, List[str]]] = None,
        obs_key_shapes: Optional[Dict[str, Tuple[int]]] = None,
    ):
        if not ((obs_config is None) == (obs_key_shapes is None)):
            raise ValueError(
                "obs_config and obs_key_shapes must either both be None or both be not None.\n"
                "Joint encoding is both are provided, otherwise separate encoding is used."
            )
    
        if obs_config is None:
            # Separate encoding, each encoder produces a separate feature encoding
            class ObservationEncoder(BaseSensoryEncoder):
                def __init__(self):
                    super().__init__()
                    self.obs_encoders = nn.ModuleList(obs_encoders)
                    self.modalities_ = modalities

                def forward(self, obs_dicts: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
                    return [
                        obs_encoder(obs_dict) 
                        for obs_encoder, obs_dict in zip(self.obs_encoders, obs_dicts)
                    ]
                
                def output_feature_dim(self) -> List[Dict[str, int]]:
                    output_feature_dim = [obs_enc.output_feature_dim() for obs_enc in obs_encoders]
                    return output_feature_dim
                
                def modalities(self) -> List[str]:
                    return self.modalities_

        else:
            # Joint encoding, all encoders produce a joint feature encoding
            class ObservationEncoder(BaseSensoryEncoder):
                def __init__(self):
                    super().__init__()
                    self.obs_encoders = nn.ModuleList(obs_encoders)
                    self.modalities_ = modalities

                def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                    features = dict()
                    for obs_encoder in self.obs_encoders:
                        features.update(obs_encoder(obs_dict))
                    for key in obs_config['robot_state']:
                        features[key] = obs_dict[key]
                    return features
                
                def output_feature_dim(self) -> Dict[str, int]:
                    output_feature_dim = dict()
                    for obs_enc in obs_encoders:
                        output_feature_dim.update(obs_enc.output_feature_dim())
                    for key in obs_config['robot_state']:
                        output_feature_dim[key] = obs_key_shapes[key][-1]
                    return output_feature_dim
                
                def modalities(self) -> List[str]:
                    return self.modalities_

        return ObservationEncoder()

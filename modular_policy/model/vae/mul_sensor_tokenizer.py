import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import einops

from modular_policy.model.vae.model import VAE
from modular_policy.model.common.normalizer import LinearNormalizer

class MultiSensorTokenizer(nn.Module):
    def __init__(self,
                 horizon: int,
                 shape_meta: dict,
                 n_latent_dims: int = 512,
                 sensor_feature_dims: Dict[str, int] = None,
                 enable_sensor_fusion: bool = True,
                 use_conv_encoder: bool = True,
                 **vae_kwargs):
        super().__init__()
        
        self.at = VAE(
            horizon=horizon,
            shape_meta=shape_meta,
            n_latent_dims=n_latent_dims,
            use_conv_encoder=use_conv_encoder,
            **vae_kwargs
        )
        
        self.enable_sensor_fusion = enable_sensor_fusion
        self.sensor_feature_dims = sensor_feature_dims or {}
        
        if enable_sensor_fusion and sensor_feature_dims:
            self.sensor_refiners = nn.ModuleDict()
            action_dim = shape_meta['action']['shape'][0]
            
            for sensor_type, feat_dim in sensor_feature_dims.items():
                self.sensor_refiners[sensor_type] = nn.Sequential(
                    nn.Linear(feat_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, action_dim)
                )
    
    def encode_to_latent(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """Encode action chunk to latent space using VAE encoder."""
        preprocessed = self.at.preprocess(action_chunk / self.at.act_scale)
        latent = self.at.encoder(preprocessed)
        
        if self.at.use_vq:
            latent, _, _ = self.at.quant_state_with_vq(latent)
        else:
            latent, _ = self.at.quant_state_without_vq(latent)
            latent = self.at.postprocess_quant_state_without_vq(latent)
        
        return latent
    
    def decode_from_latent(self, latent: torch.Tensor,
                          sensor_features: Dict[str, torch.Tensor] = None) -> torch.Tensor:
        """Decode latent to actions with optional sensor refinement."""
        # Use RDP's decoder for base actions
        base_actions = self.at.get_action_from_latent(latent)
        
        if not self.enable_sensor_fusion or not sensor_features:
            return base_actions
        
        # Add sensor-specific refinements
        sensor_refinements = []
        for sensor_type, features in sensor_features.items():
            if sensor_type in self.sensor_refiners:
                refinement = self.sensor_refiners[sensor_type](features)
                sensor_refinements.append(refinement)
        
        if sensor_refinements:
            # Average refinements and add to base actions
            avg_refinement = torch.stack(sensor_refinements).mean(dim=0)
            refined_actions = base_actions + 0.1 * avg_refinement.unsqueeze(1)
            return refined_actions
        
        return base_actions
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Use RDP's loss computation."""
        return self.at.compute_loss_and_metric(batch)
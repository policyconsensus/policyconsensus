import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from modular_policy.model.perception.sensory_encoder import BaseSensoryEncoder
from modular_policy.model.common.normalizer import LinearNormalizer
from modular_policy.policy.base_policy import BasePolicy
from modular_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from modular_policy.common.pytorch_util import dict_apply

from typing import Dict, Tuple, Optional, List

"""
formulation: A ~ P(A | O)
"""
class DiffusionUnetPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        noise_scheduler: DDPMScheduler,
        obs_encoders: List[BaseSensoryEncoder],
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        diffusion_step_embed_dim: int = 256,
        down_dims: Tuple[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        # parameters passed to step
        **kwargs
    ):
        super().__init__()

        modalities = ['robot_state'] + list(set([
            modality 
            for obs_encoder in obs_encoders 
            for modality in obs_encoder.modalities()
        ]))

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {key: [] for key in modalities}
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr['type']
            if type in modalities:
                obs_config[type].append(key)
                obs_ports.append(key)

        # create observation encoder
        obs_encoder = self.create_observation_encoder(
            obs_encoders=obs_encoders,
            modalities=modalities,
            obs_config=obs_config,
            obs_key_shapes=obs_key_shapes
        )

        # create diffusion model
        obs_feature_dim = sum(obs_encoder.output_feature_dim().values())
        global_cond_dim = obs_feature_dim * n_obs_steps
        
        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        num_diffusion_params = sum(p.numel() for p in self.model.parameters())
        num_vision_params = sum(p.numel() for p in self.obs_encoder.parameters())
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"    {num_diffusion_params} diffusion params\n"
            f"    {num_vision_params} perception params\n"
        )


    def get_noise_prediction_network(self) -> nn.Module:
        return self.model
    

    def get_observation_encoder(self) -> BaseSensoryEncoder:
        return self.obs_encoder
    

    def get_observation_modalities(self) -> List[str]:
        return self.modalities
    

    def get_observation_ports(self) -> List[str]:
        return self.obs_ports


    def get_policy_name(self) -> str:
        base_name = 'dp_unet_'
        for modality in self.modalities:
            if modality != 'robot_state':
                base_name += modality + '|'
        return base_name[:-1]
    

    def create_dummy_observation(self, 
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )

    
    def conditional_sample(self, 
        global_cond,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs
    ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=(len(global_cond), self.horizon, self.action_dim),
            dtype=global_cond.dtype,
            device=global_cond.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            trajectory = scheduler.step(
                model(trajectory, t, global_cond=global_cond), 
                t, trajectory, 
                generator=generator,
                **kwargs
            ).prev_sample
        return trajectory
    

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        Da = self.action_dim
        To = self.n_obs_steps

        # run sampling
        features = self.obs_encoder(
            dict_apply(
                nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:])
            )
        )
        features = dict_apply(features, lambda x: x.reshape(B, -1))
        global_cond = torch.cat(list(features.values()), dim=-1)
        nsample = self.conditional_sample(
            global_cond=global_cond,
            **self.kwargs
        )
        
        # unnormalize prediction
        action_pred = self.normalizer['action'].unnormalize(nsample[...,:Da])

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result
    

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_loss(self, batch):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        trajectory = self.normalizer['action'].normalize(batch['action'])
        batch_size = trajectory.shape[0]

        # global conditioning
        features = self.obs_encoder(
            dict_apply(
                nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:])
            )
        )
        features = dict_apply(features, lambda x: x.reshape(batch_size, -1))
        global_cond = torch.cat(list(features.values()), dim=-1)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        assert pred_type == 'epsilon', "Only epsilon prediction is supported"

        # noise prediction loss
        loss = F.mse_loss(pred, noise)

        return loss

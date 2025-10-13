import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from modular_policy.policy.base_policy import BasePolicy
from modular_policy.model.perception.sensory_encoder import BaseSensoryEncoder
from modular_policy.model.common.normalizer import LinearNormalizer
from modular_policy.common.pytorch_util import dict_apply

from typing import List, Dict, Optional

"""
formulation: A ~ Π wi * Pi(A | Oi) via energy-based composition
"""
class DiffusionCompositionalPolicy(BasePolicy):
    def __init__(self,
        shape_meta: Dict,
        noise_scheduler: DDPMScheduler,
        modular_policies: List[BasePolicy],
        policy_weights: List[float],
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: Optional[int] = None,
        # parameters passed to step
        **kwargs
    ):
        super().__init__()

        modalities = list(set(['robot_state', *[
            modality 
            for policy in modular_policies
            for modality in policy.get_observation_modalities()
        ]]))

        # parse shape / obs config
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_ports = list(set([
            port 
            for policy in modular_policies
            for port in policy.get_observation_ports()
        ]))

        # create observation encoder
        obs_encoder = self.create_observation_encoder(
            obs_encoders=[policy.get_observation_encoder() for policy in modular_policies],
            modalities=modalities
        )

        # create normalizer and unnormalizer
        def normalization(obs_dict: Dict[str, torch.Tensor]) -> List[Dict[str, torch.Tensor]]:
            # normalize each observation w.r.t. each policy
            nobs_list = []
            for policy in modular_policies:
                this_obs = {port: obs_dict[port] for port in policy.get_observation_ports()}
                this_nobs = policy.normalizer.normalize(this_obs)
                nobs_list.append(this_nobs)
            return nobs_list
        # check if action normalizations are all the same
        action_scales = [policy.normalizer['action'].params_dict['scale'] for policy in modular_policies]
        action_offsets = [policy.normalizer['action'].params_dict['offset'] for policy in modular_policies]
        if not (
            all([torch.all(scale == action_scales[0]) for scale in action_scales]) and
            all([torch.all(offset == action_offsets[0]) for offset in action_offsets])
        ):
            raise ValueError("Action normalizations are not the same")
        def unnormalization(action: torch.Tensor):
            # all policies have the same action normalizer
            return modular_policies[0].normalizer['action'].unnormalize(action)

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps

        for weight in policy_weights:
            assert weight > 0, "policy weights must be positive"

        # attr assignment
        self.normalization = normalization
        self.unnormalization = unnormalization
        self.modalities = modalities
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.policy_weights = policy_weights
        self.modular_policies = nn.ModuleList(modular_policies)
        self.num_inference_steps = num_inference_steps
        self.noise_scheduler = noise_scheduler
        self.horizon = horizon
        self.obs_feature_dims = obs_encoder.output_feature_dim()
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        num_diffusion_params = sum(sum(
            p.numel() for p in policy.get_noise_prediction_network().parameters()
        ) for policy in modular_policies)
        num_vision_params = sum(sum(
            p.numel() for p in policy.get_observation_encoder().parameters()
        ) for policy in modular_policies)
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"    {num_diffusion_params} diffusion params\n"
            f"    {num_vision_params} perception params\n"
        )


    def get_noise_prediction_network(self) -> nn.ModuleList:
        return [policy.get_noise_prediction_network() for policy in self.modular_policies]
    

    def get_observation_encoder(self) -> BaseSensoryEncoder:
        return self.obs_encoder
    

    def get_observation_modalities(self) -> List[str]:
        return self.modalities
    

    def get_observation_ports(self) -> List[str]:
        return self.obs_ports
    

    def get_policy_name(self) -> str:
        base_name = 'dp_compositional_'
        for modality in self.modalities:
            if modality != 'robot_state':
                base_name += modality + '|'
        return base_name[:-1]


    def conditional_sample(self,
        global_conds: List[torch.Tensor],
        generator: Optional[torch.Generator] = None,
        # keyword arguments to scheduler.step
        **kwargs
    ) -> torch.Tensor:
        models = [policy.get_noise_prediction_network() for policy in self.modular_policies]
        assert len(models) == len(global_conds) == len(self.policy_weights)
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=(len(global_conds[0]), self.horizon, self.action_dim),
            dtype=global_conds[0].dtype,
            device=global_conds[0].device,
            generator=generator
        )

        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:

            noise = sum(
                weight * model(trajectory, t, global_cond=cond)
                for weight, model, cond in zip(self.policy_weights, models, global_conds)
            )

            trajectory = scheduler.step(
                noise, t, trajectory,
                generator=generator,
                **kwargs
            ).prev_sample

        return trajectory
    

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs_list = self.normalization(obs_dict)
        value = next(iter(nobs_list[0].values()))
        B, To = value.shape[:2]
        Da = self.action_dim
        To = self.n_obs_steps
        
        # run sampling
        nsample = self.conditional_sample(
            global_conds=[
                torch.cat([v.reshape(B, -1) for v in nobs_features.values()], dim=1) 
                for nobs_features in self.obs_encoder([
                    dict_apply(
                        nobs, 
                        lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:])
                    ) for nobs in nobs_list
                ])
            ],
            **self.kwargs
        )

        # unnormalize prediction
        action_pred = self.unnormalization(nsample[..., :Da])

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start : end]

        return {
            'action': action,
            'action_pred': action_pred
        }


    def set_normalizer(self, normalizer: List[LinearNormalizer]):
        # check if action normalizations are all the same
        action_scales = [norm['action'].params_dict['scale'] for norm in normalizer if 'action' in norm]
        action_offsets = [norm['action'].params_dict['offset'] for norm in normalizer if 'action' in norm]
        if not (
            all([torch.all(scale == action_scales[0]) for scale in action_scales]) and
            all([torch.all(offset == action_offsets[0]) for offset in action_offsets])
        ):
            raise ValueError("Action normalizations are not the same")
        
        for policy, norm in zip(self.modular_policies, normalizer):
            policy.set_normalizer(norm)

import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import matplotlib.pyplot as plt
from modular_policy.model.perception.sensory_encoder import BaseSensoryEncoder
from modular_policy.model.common.normalizer import LinearNormalizer
from modular_policy.policy.dp_unets_base import DiffusionUnetsBase
from modular_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from modular_policy.common.pytorch_util import dict_apply

from typing import List, Dict, Optional, Tuple

"""
formulation: A ~ Π Pi(A | Mi) via energy-based composition and Mi is ith sensory modality
"""
class DiffusionUnetsPolicy(DiffusionUnetsBase):
    def __init__(self, 
        shape_meta: dict,
        num_modules: int,
        noise_scheduler: DDPMScheduler,
        obs_encoders: List[BaseSensoryEncoder],
        horizon, 
        n_action_steps, 
        n_obs_steps,
        num_inference_steps: Optional[int] = None,
        diffusion_step_embed_dim: int = 128,
        down_dims: Tuple[int] = (128,256,512),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        composition_strategy: str = "soft_gating",  # Options: soft_gating, hard_routing, topk_moe
        topk: int = 2,  # Only used for top-k MoE
        # parameters passed to step
        **kwargs
    ):
        super().__init__()

        # modalities = ['robot_state'] + list(set([
        #     modality 
        #     for obs_encoder in obs_encoders 
        #     for modality in obs_encoder.modalities()
        # ]))
        preferred_order = ['rgb', 'depth', 'pointcloud', 'dino_pointcloud']
        modalities_set = {
            modality
            for encoder in obs_encoders
            for modality in encoder.modalities()
        }

        # Split into preferred and remaining
        ordered = [m for m in preferred_order if m in modalities_set]
        remaining = sorted(modalities_set - set(ordered))

        # Final list with robot_state always first
        modalities = ['robot_state', *ordered, *remaining]


        # parse shape_meta
        action_shape = shape_meta['action']['shape']
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
            obs_key_shapes=obs_key_shapes,
        )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_feature_dim()
        models = nn.ModuleList()
        for modality in modalities:
            if modality == "robot_state": 
                continue 
            fdim = sum(obs_feature_dim[key] for key in (obs_config[modality] + obs_config['robot_state']))
            models.extend([
                ConditionalUnet1D(
                    input_dim=action_dim,
                    local_cond_dim=None,
                    global_cond_dim=fdim * n_obs_steps,
                    diffusion_step_embed_dim=diffusion_step_embed_dim,
                    down_dims=down_dims,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale
                )
                for _ in range(num_modules)
            ])


        # create weight prediction network, output numbers between -1 and 1
        obs_feature_dim = sum(obs_feature_dim.values())
        global_cond_dim = obs_feature_dim * n_obs_steps
        weight_predictor = nn.Sequential(
            nn.Linear(global_cond_dim, global_cond_dim),
            nn.ReLU(),
            nn.Linear(global_cond_dim, len(models)),
            # nn.Tanh()
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_config = obs_config
        self.obs_encoder = obs_encoder
        self.models = models
        self.weight_predictor = weight_predictor
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_modules = num_modules
        self.composition_strategy = composition_strategy
        self.topk = topk
        assert composition_strategy in ["soft_gating", "hard_routing", "topk_moe"]

        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        num_diffusion_params = sum(p.numel() for p in self.models.parameters())
        num_vision_params = sum(p.numel() for p in self.obs_encoder.parameters())
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"    {num_diffusion_params} diffusion params\n"
            f"    {num_vision_params} perception params\n"
        )


    def get_noise_prediction_network(self) -> nn.Module:
        return self.models
    

    def get_observation_encoder(self) -> BaseSensoryEncoder:
        return self.obs_encoder
    

    def get_observation_modalities(self) -> List[str]:
        return self.modalities
    
    
    def get_observation_ports(self) -> List[str]:
        return self.obs_ports
    

    def get_policy_name(self) -> str:
        base_name = 'dp_unets_spec_'
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
        global_conds: List[torch.Tensor],
        weights: torch.Tensor,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs
    ):
 
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=(len(global_conds[0]), self.horizon, self.action_dim),
            dtype=global_conds[0].dtype,
            device=global_conds[0].device,
            generator=generator
        )
        B = trajectory.shape[0]

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        for t in scheduler.timesteps:
            if self.composition_strategy == "soft_gating":
                norm_weights = F.softmax(weights, dim=0)
                pred = sum(
                    norm_weights[i][:, None, None] * model(trajectory, t, global_cond=global_conds[i//self.num_modules])
                    for i, model in enumerate(self.models)
                )

            elif self.composition_strategy == "hard_routing":
                idx = torch.argmax(weights, dim=0)  # (B,)
                pred = torch.stack([
                    self.models[i](trajectory[b:b+1], t, global_cond=global_conds[i//self.num_modules][b:b+1])
                    for b, i in enumerate(idx)
                ]).squeeze(1)

            elif self.composition_strategy == "topk_moe":
                # topk_vals, topk_idx = torch.topk(weights, k=2, dim=0)  # (k, B)
                # norm_weights = F.softmax(topk_vals, dim=0)             # (k, B)
                # pred = torch.zeros_like(trajectory)

                # for i in range(len(self.models)):
                #     # mask: shape (k, B), True where expert i is selected
                #     mask = topk_idx == i  # bool tensor

                #     if not mask.any():
                #         continue

                #     b_indices = torch.nonzero(mask, as_tuple=False)[:, 1]  # batch indices
                #     weights_selected = norm_weights[mask]  # shape (num_selected,)
                #     traj_batch = trajectory[b_indices]
                #     cond_batch = global_conds[i // self.num_modules][b_indices]

                #     out = self.models[i](traj_batch, t, global_cond=cond_batch)
                #     pred[b_indices] += weights_selected[:, None] * out
                k = min(self.topk, len(self.models))
                topk_vals, topk_idx = torch.topk(weights, k=k, dim=0)  # (k, B)
                norm_weights = F.softmax(topk_vals, dim=0)  # (k, B)
                pred = torch.zeros_like(trajectory)

                for j in range(k):
                    idx = topk_idx[j]  # (B,)
                    w = norm_weights[j]  # (B,)
                    for b in range(B):
                        i = idx[b].item()
                        pred[b] += w[b] * self.models[i](trajectory[b:b+1], t, global_cond=global_conds[i//self.num_modules][b:b+1]).squeeze(0)
            else:
                raise ValueError(f"Unknown strategy: {self.composition_strategy}")

            trajectory = scheduler.step(pred, t, trajectory, generator=generator, **kwargs).prev_sample                
                
                
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
        global_conds = [
            torch.cat([features[key] for key in (self.obs_config[modality]+self.obs_config["robot_state"])], dim=-1)
            for modality in self.modalities if modality != "robot_state"
        ]


        global_cond = torch.cat(list(features.values()), dim=-1)
        weights = self.weight_predictor(global_cond).transpose(0, 1)

        # self.log_weights(weights)
        # print(
        #     f"weights: \n{weights}\n"
        # )

        # if self.num_weights_log % 10 == 0:
        # self.plot_weights_log(
        #     save_path='test.png',
        #     method='lineplot'
        # )

        nsample = self.conditional_sample(
            global_conds=global_conds,
            weights=weights,
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
            'action_pred': action_pred,
            'weights': weights.permute(1, 0),
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
        
        global_conds = [
            torch.cat([features[key] for key in (self.obs_config[modality]+self.obs_config["robot_state"])], dim=-1)
            for modality in self.modalities if modality != "robot_state"
        ]

        global_cond = torch.cat(list(features.values()), dim=-1)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=trajectory.device
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        weights = self.weight_predictor(global_cond).transpose(0, 1)

        # self.log_weights(weights)
        # if self.num_weights_log % 500 == 0:
        #     self.plot_weights_log(
        #         save_path='test.png',
        #         method='lineplot'
        #     )

        # Predict the noise residual
        # pred = sum(
        #     w[:,None,None] * model(noisy_trajectory, timesteps, global_cond=global_cond)
        #     for w, model, global_cond in zip(weights, self.models, global_conds)
        # )
        if self.composition_strategy == "soft_gating":
            norm_weights = F.softmax(weights, dim=0)
            pred = sum(
                norm_weights[i][:, None, None] * model(noisy_trajectory, timesteps, global_cond=global_conds[i//self.num_modules])
                for i, model in enumerate(self.models)
            )

        elif self.composition_strategy == "hard_routing":
            idx = torch.argmax(weights, dim=0)
            pred = torch.stack([
                self.models[i](noisy_trajectory[b:b+1], timesteps[b:b+1], global_cond=global_conds[i//self.num_modules][b:b+1])
                for b, i in enumerate(idx)
            ]).squeeze(1)

        elif self.composition_strategy == "topk_moe":
            k = min(self.topk, len(self.models))
            topk_vals, topk_idx = torch.topk(weights, k=k, dim=0)
            norm_weights = F.softmax(topk_vals, dim=0)
            pred = torch.zeros_like(noisy_trajectory)

            for j in range(k):
                idx = topk_idx[j]
                w = norm_weights[j]
                for b in range(batch_size):
                    i = idx[b].item()
                    pred[b] += w[b] * self.models[i](noisy_trajectory[b:b+1], timesteps[b:b+1], global_cond=global_conds[i//self.num_modules][b:b+1]).squeeze(0)
        else:
            raise ValueError(f"Unknown strategy: {self.composition_strategy}")

        pred_type = self.noise_scheduler.config.prediction_type 
        assert pred_type == 'epsilon', "Only epsilon prediction is supported"

        # noise prediction loss
        loss = F.mse_loss(pred, noise)

        return loss
    
    
    def adapt(self, method: str = 'weight_predictor+obs_encoder'):
        if method == 'weight_predictor':
            # freeze all parameters except for the weight predictor
            self.eval()
            for model in self.models:
                model.requires_grad_(False)
            self.obs_encoder.requires_grad_(False)
            self.weight_predictor.train()
        elif method == 'weight_predictor+obs_encoder':
            # freeze all parameters except for the weight predictor and obs_encoder
            self.eval()
            for model in self.models:
                model.requires_grad_(False)
            self.obs_encoder.train()
            self.weight_predictor.train()
        elif method == 'full':
            # unfreeze all parameters
            self.train()
        else:
            raise ValueError(f"Unknown adaptation method: {method}")

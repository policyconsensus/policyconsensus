import torch
import numpy as np
import copy
import os
import zarr

from modular_policy.common.pytorch_util import dict_apply
from modular_policy.common.replay_buffer import ReplayBuffer
from modular_policy.common.seq_sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from modular_policy.model.common.normalizer import LinearNormalizer
from modular_policy.dataset.base_dataset import BaseDataset

from typing import Dict, Optional, List



class RlbenchDataset(BaseDataset):
    def __init__(
        self,
        zarr_path: str,
        obs_keys: List[str],
        action_key: str = 'action',
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0, 
        seed: int = 42,
        val_ratio: float = 0.0,
        max_train_episodes: Optional[int] = None,
        image_scale: float = 255.0,
    ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[action_key, *obs_keys],
        )
        if 'fused_dino' in self.replay_buffer:
            pca_path = os.path.join(zarr_path, 'pca_dino_3d.pkl')
            self.apply_shared_pca(
                zarr_path=zarr_path,
                src_key='fused_dino', 
                pca_path=pca_path
            )

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        self.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask
        )

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_keys = obs_keys
        self.action_key = action_key
        self.image_scale = image_scale


    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.seq_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set
    

    def get_normalizer(self, mode='limits', **kwargs):
        data = {'action': self.replay_buffer[self.action_key]}
        
        # Apply image scaling to observation data for normalizer fitting
        for k in self.obs_keys:
            obs_data = self.replay_buffer[k]
            if 'rgb' in k and self.image_scale != 255.0:
                # Convert from [0, 255] to desired scale
                obs_data = obs_data.astype(np.float32) * (self.image_scale / 255.0)
            data[k] = obs_data
            
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer
    

    def __len__(self):
        return len(self.seq_sampler)
    

    def _sample_to_data(self, sample):
        obs_data = {}
        for k in self.obs_keys:
            obs_data[k] = sample[k].astype(np.float32)
            # Apply image scaling for RGB observations
            if 'rgb' in k and self.image_scale != 255.0:
                # Convert from [0, 255] to desired scale
                obs_data[k] = obs_data[k] * (self.image_scale / 255.0)
        
        data = {
            'obs': obs_data,
            'action': sample[self.action_key].astype(np.float32),
        }
        return data
    

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.seq_sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, lambda x: torch.from_numpy(x))
        return torch_data

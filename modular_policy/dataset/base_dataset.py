from typing import Dict

import numpy as np
import os
import tqdm
import zarr
from PIL import Image
from matplotlib import pyplot as plt
import open3d as o3d
import time
import torch
import torch.nn
from modular_policy.model.common.normalizer import LinearNormalizer

class BaseDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseDataset':
        return BaseDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()

    def replay(self, obs_key: str, type: str, dt: float = 0.1, sample_every: int = 1):
        try:
            if type == 'pointcloud':
                o3dvis = o3d.visualization.Visualizer()
                o3dvis.create_window()
                pcd = None

            for idx in tqdm.tqdm(range(len(self)), desc=f'Replay {obs_key}'):
                sample = self.seq_sampler.sample_sequence(idx)
                obs = self._sample_to_data(sample)['obs'][obs_key][0]
                if type == 'rgb' and idx % sample_every == 0:
                    obs = np.moveaxis(obs, 0, -1)
                    Image.fromarray(obs.astype(np.uint8)).save('rgb.png')
                    time.sleep(dt)
                elif type == 'depth' and idx % sample_every == 0:
                    obs = (obs - obs.min()) / (obs.max() - obs.min())   # (1, H, W)
                    plt.imsave('depth.png', obs[0])
                    time.sleep(dt)
                elif type == 'pointcloud' and idx % sample_every == 0:
                    has_color = (obs.shape[-1] == 6)
                    if pcd is None:
                        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obs[:, :3]))
                        if has_color:
                            pcd.colors = o3d.utility.Vector3dVector(obs[:, 3:])
                        o3dvis.add_geometry(pcd)
                    else:
                        pcd.points = o3d.utility.Vector3dVector(obs[:, :3])
                        if has_color:
                            pcd.colors = o3d.utility.Vector3dVector(obs[:, 3:])
                        o3dvis.update_geometry(pcd)
                    o3dvis.poll_events()
                    o3dvis.update_renderer()
                    time.sleep(dt)
        except KeyboardInterrupt as e:
            print(f"An error occurred: {e}")
        finally:
            if os.path.exists('rgb.png'):
                os.remove('rgb.png')
            if os.path.exists('depth.png'):
                os.remove('depth.png')
            if 'o3dvis' in locals():
                o3dvis.destroy_window()

    def apply_shared_pca(self, zarr_path, src_key='fused_dino', pca_path='pca_dino_3d.pkl'):
        import joblib
        import os
        from sklearn.decomposition import PCA
        
        data = self.replay_buffer[src_key]
        
        store = zarr.open(zarr_path, mode='a')  
        data_group = store['data']

        # Skip if already processed (e.g., shape is [N, M, 6])
        if data.ndim == 3 and data.shape[-1] == 6:
            print(f"[PCA] '{src_key}' already reduced to 6D. Skipping PCA.")
            return
        
        # Sanity check for expected shape before PCA
        if data.ndim != 3 or data.shape[-1] != 387:
            raise ValueError(
                f"[PCA Error] Expected '{src_key}' to have shape (N, M, 387), "
                f"but got shape {data.shape}. Cannot apply PCA."
            )

        if os.path.exists(pca_path):
            print(f"Loading existing PCA from {pca_path}")
            pca = joblib.load(pca_path)
        else:
            print(f"PCA file not found, fitting new PCA and saving to {pca_path}")
            feats = self.replay_buffer[src_key][..., 3:].reshape(-1, 384)
            pca = PCA(n_components=3).fit(feats)
            joblib.dump(pca, pca_path)

        raw_key = src_key + '_raw'
        if raw_key not in data_group:
            print(f"[PCA] Writing raw '{src_key}' to '{raw_key}' on disk (not kept in memory)")
            data_group.create_dataset(
                raw_key,
                data=data,
                compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=2)
            )

        xyz = self.replay_buffer[src_key][..., :3]
        feats = self.replay_buffer[src_key][..., 3:]
        reduced_feats = pca.transform(feats.reshape(-1, 384)).reshape(feats.shape[0], feats.shape[1], 3)
        self.replay_buffer.data[src_key] = np.concatenate([xyz, reduced_feats], axis=-1)
        
        if src_key in data_group:
            del data_group[src_key]
        data_group.create_dataset(
            src_key,
            data=self.replay_buffer['fused_dino'],
            compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=2)
        )
        
        if raw_key in self.replay_buffer.data:
            del self.replay_buffer.data[raw_key]

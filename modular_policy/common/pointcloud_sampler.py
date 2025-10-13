import torch
import numpy as np
from pytorch3d.ops import sample_farthest_points

from typing import Union


def uniform_sampling(
    points: Union[torch.Tensor,  np.ndarray],
    num_points: int
) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(points, torch.Tensor):
        indices = torch.randperm(points.shape[-2])[:num_points]
    else:
        indices = np.random.permutation(points.shape[-2])[:num_points]
    return points[..., indices, :]


def farthest_point_sampling(
    points: Union[torch.Tensor,  np.ndarray],
    num_points: int,
    random_start: bool = False
) -> Union[torch.Tensor, np.ndarray]:
    if points.shape[-2] < num_points:
        raise ValueError(
            f"Number of points in the point cloud ({points.shape[-2]}) "
            f"is less than the number of points to sample ({num_points})."
        )
    elif points.shape[-2] == num_points:
        return points

    unsqueezed = False
    if isinstance(points, torch.Tensor):
        if points.ndim == 2:
            points = points[None]
            unsqueezed = True
        _, subsampled_idx = sample_farthest_points(
            points[..., :3], 
            K=num_points, random_start_point=random_start
        )
        subsampled_pcd = torch.gather(
            input=points,
            dim=1,
            index=subsampled_idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        )
    else:
        if points.ndim == 2:
            points = points[np.newaxis]
            unsqueezed = True
        points = torch.from_numpy(points)
        _, subsampled_idx = sample_farthest_points(
            points[..., :3], 
            K=num_points, random_start_point=random_start
        )
        subsampled_pcd = torch.gather(
            input=points,
            dim=1,
            index=subsampled_idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        )
        subsampled_pcd = subsampled_pcd.cpu().numpy()
        points = points.cpu().numpy()
    if unsqueezed:
        points = points.squeeze(0)
        subsampled_pcd = subsampled_pcd.squeeze(0)
    return subsampled_pcd


def pointcloud_subsampling(
    points: Union[torch.Tensor,  np.ndarray],
    num_points: int,
    method: str = 'fps'
) -> Union[torch.Tensor, np.ndarray]:
    # points: (*, N, d)
    # num_points: int
    # method: str

    if method == 'uniform':
        return uniform_sampling(points, num_points)
    elif method == 'fps':
        return farthest_point_sampling(points, num_points)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
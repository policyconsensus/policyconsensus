import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
from typing import Optional


def load_dino_model(name: str = 'dinov2_vits14'):
    dino_model = torch.hub.load('facebookresearch/dinov2', name)
    dino_model.eval()
    return dino_model


def compute_dino_features(
    dino_model,
    images: torch.Tensor,
):
    # images: (B, 3, H, W)
    # return:
    #   feat_images:    (B, D, H, W)
    #   feat_patches:   (B, D, Hp, Wp)

    B, C, H, W = images.shape
    P = dino_model.patch_size
    D = dino_model.num_features
    
    assert C == 3, "Input images must have 3 channels (RGB)."

    # resize to multiple of patch size
    Hp = round(H / P) * P
    Wp = round(W / P) * P
    transform = transforms.Compose([
        transforms.Resize((Hp, Wp), antialias=True),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    images = transform(images)

    # compute dino features
    with torch.inference_mode():
        features = dino_model.forward_features(images)
        feat_patches = features['x_norm_patchtokens'].view(
            B, Hp // P, Wp // P, D).permute(0, 3, 1, 2)
        feat_images = F.interpolate(
            feat_patches, 
            size=(H, W), mode='bilinear', align_corners=False
        )  # (B, D, H, W)

    return feat_images, feat_patches


@torch.inference_mode()
def fuse_descriptor_field(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    dino_model: Optional[torch.nn.Module] = None,
    pca_dim: Optional[int] = None,
):
    # rgb: (N, 3, H, W)
    # depth: (N, 1, H, W)
    # extrinsics: (N, 4, 4)
    # intrinsics: (N, 3, 3)
    # dino_model: optional, if None, will load the default DINOv2 model
    # pca_dim: optional, if provided, will reduce the DINO features to this dimension with PCA
    # return: (N * H * W, 3 + D) where D is the feature dimension

    N, _, H, W = rgb.shape

    if dino_model is None:
        dino_model = load_dino_model('dinov2_vits14').to(rgb.device)
    D = dino_model.num_features

    if pca_dim is not None:
        assert 1 <= pca_dim <= D, f"PCA dimension must be between 1 and {D}, got {pca_dim}."
    
    # """
    # create raw point cloud
    # """
    u, v = torch.meshgrid(
        torch.arange(H, device=depth.device),
        torch.arange(W, device=depth.device),
        indexing='ij'
    )
    u = u.float()[None].expand(N, -1, -1)  # (N, H, W)
    v = v.float()[None].expand(N, -1, -1)

    z = depth[:, 0]  # (N, H, W)
    fx, fy, cx, cy = [intrinsics[:, i, j][:, None, None] for (i, j) in [(0, 0), (1, 1), (0, 2), (1, 2)]]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = torch.stack((x, y, z), dim=-1) # (N, H, W, 3)

    # to world coordinates
    cam_poses = extrinsics.inverse()  # (N, 4, 4)
    points = points.view(N, H * W, 3).transpose(1, 2)  # (N, 3, H*W)
    points = torch.bmm(cam_poses[:, :3, :3], points) + cam_poses[:, :3, 3:]   # (N, 3, H*W)
    points = points.transpose(1, 2)  # (N, H*W, 3)

    # """
    # compute features
    # """
    feats, _ = compute_dino_features(dino_model, rgb)               # (N, D, H, W)
    feats = feats.permute(0, 2, 3, 1).view(N, H * W, D)
    if pca_dim is not None:
        feats = feats.view(N * H * W, D)
        feats = pca_dim_reduction(feats.cpu().numpy(), n_comp=pca_dim)    # (N * H * W, pca_dim)
        feats = torch.from_numpy(feats).to(rgb.device).view(
            N, H * W, pca_dim)
        D = pca_dim

    # """
    # combine points and features and rule out invalid points
    # """
    valid_mask = ((depth[:, 0] > 0) & torch.isfinite(depth[:, 0])).view(N, H * W)
    field = torch.concat((points[valid_mask], feats[valid_mask]), dim=-1)   # (N * H * W, 3 + D)
    return field


def pca_dim_reduction(feat: np.ndarray, n_comp: int):
    # feat: (B, D)
    # return: (B, n_comp)
    pca = PCA(n_components=n_comp)
    feat_reduced = pca.fit_transform(feat)
    return feat_reduced

"""
Evaluation Visualization Module

This module contains visualization functions for policy evaluation.
Functions handle tactile sensor and camera grid visualization.
"""

import numpy as np
import cv2


def get_tactile_viz(obs, tactile_scale_factor=25):
    """
    Create visualization for tactile sensor data.

    Args:
        obs: Dictionary containing 'left_tactile' and 'right_tactile' observations
        tactile_scale_factor: Scale factor for tactile display size (default: 25)

    Returns:
        tactile_viz: OpenCV image showing tactile data with VIRIDIS colormap
    """
    tactile_viz = None
    assert 'left_tactile' in obs and 'right_tactile' in obs
    # Get most recent tactile data
    left_tactile = obs['left_tactile'][-1][0]
    right_tactile = obs['right_tactile'][-1][0]

    # Scale to 0-255 (3D-ViTac approach)
    left_scaled = (left_tactile * 255).astype(np.uint8)
    right_scaled = (right_tactile * 255).astype(np.uint8)

    # Apply VIRIDIS colormap (3D-ViTac style)
    left_colored = cv2.applyColorMap(left_scaled, cv2.COLORMAP_VIRIDIS)
    right_colored = cv2.applyColorMap(right_scaled, cv2.COLORMAP_VIRIDIS)

    # Use configurable scale factor for tactile display size
    left_resized = cv2.resize(left_colored,
                            (left_tactile.shape[1] * tactile_scale_factor, left_tactile.shape[0] * tactile_scale_factor),
                            interpolation=cv2.INTER_NEAREST)
    right_resized = cv2.resize(right_colored,
                            (right_tactile.shape[1] * tactile_scale_factor, right_tactile.shape[0] * tactile_scale_factor),
                            interpolation=cv2.INTER_NEAREST)

    # Add labels (3D-ViTac style)
    cv2.putText(left_resized, 'Left Tactile', (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(right_resized, 'Right Tactile', (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Stack tactile visualizations vertically (left on top, right on bottom)
    tactile_viz = np.vstack([left_resized, right_resized])

    return tactile_viz


def create_camera_grid_viz(obs, vis_cameras, window_width=None, window_height=None):
    """
    Create a grid visualization for selected RGB cameras.

    Args:
        obs: Dictionary containing camera observations (e.g., 'camera_0_color')
        vis_cameras: List of camera indices to display
        window_width, window_height: Optional dimensions to resize the final grid

    Returns:
        camera_grid: OpenCV image showing a grid of camera views
    """
    if not vis_cameras:
        return None

    images = []
    for cam_idx in vis_cameras:
        cam_key = f'camera_{cam_idx}_color'
        if cam_key in obs:
            images.append(obs[cam_key][-1])

    if not images:
        return None

    # Stack all images horizontally
    grid = np.hstack(images)

    if window_width and window_height:
        grid = cv2.resize(grid, (window_width, window_height))

    return grid

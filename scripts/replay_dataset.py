#!/usr/bin/env python3
"""
Dataset Replay Script for UR5e Robot

Replays recorded demonstrations from zarr datasets on real UR5e robot.
Supports both joint and end-effector control modes with raw and processed data formats.
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import numpy as np
import zarr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modular_policy.real_world.real_ur5e_env import RealUR5eEnv
from modular_policy.common.trans_utils import are_joints_close
from modular_policy.common.precise_sleep import precise_wait


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    data: Dict[str, np.ndarray]
    episode_ends: np.ndarray
    format_type: str  # 'raw' or 'processed'
    available_keys: List[str]


@dataclass
class ActionData:
    """Container for extracted action and gripper data."""
    actions: np.ndarray
    gripper_actions: Optional[np.ndarray] = None


class DatasetLoader:
    """Handles loading and parsing of zarr datasets."""
    
    @staticmethod
    def load(zarr_path: str) -> DatasetInfo:
        """Load zarr dataset and return structured information."""
        logger.info(f"Loading dataset from: {zarr_path}")
        
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Dataset not found at {zarr_path}")
        
        try:
            group = zarr.open_group(zarr_path, 'r')
            data_group = group['data']
            episode_ends = group['meta']['episode_ends'][:]
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")
        
        # Detect and normalize data format
        if 'observations' in data_group:
            logger.info("Detected raw data format")
            data_dict = DatasetLoader._normalize_raw_data(data_group)
            format_type = 'raw'
        else:
            logger.info("Detected processed data format")
            data_dict = dict(data_group)
            format_type = 'processed'
        
        available_keys = [k for k in data_dict.keys() if not k.startswith('camera')]
        logger.info(f"Dataset loaded - Episodes: {len(episode_ends)}, Keys: {available_keys}")
        
        return DatasetInfo(
            data=data_dict,
            episode_ends=episode_ends,
            format_type=format_type,
            available_keys=available_keys
        )
    
    @staticmethod
    def _normalize_raw_data(data_group) -> Dict[str, np.ndarray]:
        """Normalize raw data format to processed format structure."""
        normalized = {}
        
        # Copy observation data to top level
        if 'observations' in data_group:
            for key, value in data_group['observations'].items():
                normalized[key] = value
        
        # Copy action and other data
        for key, value in data_group.items():
            if key != 'observations':
                normalized[key] = value
                
        return normalized


class ActionExtractor:
    """Handles extraction of actions and gripper data from datasets."""
    
    @staticmethod
    def extract(dataset_info: DatasetInfo, episode_slice: slice, 
                ctrl_mode: str, joint_source: str = 'robot_joint') -> ActionData:
        """Extract actions based on control mode."""
        
        if ctrl_mode == 'joint':
            return ActionExtractor._extract_joint_actions(
                dataset_info, episode_slice, joint_source)
        else:  # eef mode
            return ActionExtractor._extract_eef_actions(
                dataset_info, episode_slice)
    
    @staticmethod
    def _extract_joint_actions(dataset_info: DatasetInfo, episode_slice: slice,
                              joint_source: str) -> ActionData:
        """Extract joint-based actions."""
        data = dataset_info.data
        
        # Validate joint source availability
        if joint_source not in data:
            available = [k for k in dataset_info.available_keys if 'joint' in k]
            raise KeyError(f"'{joint_source}' not found. Available joint keys: {available}")
        
        actions = data[joint_source][episode_slice]
        logger.info(f"Using {joint_source} - Shape: {actions.shape}")
        
        # Validate joint ranges for safety (UR5e joint limits approximately ±2π)
        if joint_source == 'robot_joint':
            joint_data_6d = actions[:, :6] if actions.shape[1] > 6 else actions
            min_vals = np.min(joint_data_6d, axis=0)
            max_vals = np.max(joint_data_6d, axis=0)
            
            # Check for reasonable joint ranges
            if np.any(np.abs(min_vals) > 3.5) or np.any(np.abs(max_vals) > 3.5):
                logger.warning(f"Joint positions may be outside normal range:")
                logger.warning(f"  Min values: {min_vals}")
                logger.warning(f"  Max values: {max_vals}")
                logger.warning("  Proceed with caution!")
        
        # Extract gripper data
        gripper_actions = ActionExtractor._extract_gripper_from_joints(
            data, episode_slice, joint_source, actions)
        
        # Handle joint dimensions
        if actions.shape[1] > 6:
            actions = actions[:, :6]
            logger.info(f"Extracted 6D joint positions - Shape: {actions.shape}")
        
        # Validate joint_action data quality and handle problematic data
        if joint_source == 'joint_action':
            if np.allclose(actions, 0):
                logger.error("joint_action data is all zeros - this would send robot to dangerous zero position!")
                logger.error("This dataset was likely collected with EEF control, not joint control.")
                logger.error("Automatically switching to robot_joint for safety.")
                logger.info("To avoid this, use: -j robot_joint or switch to: -m eef")
                
                # Automatically fall back to robot_joint for safety
                if 'robot_joint' in data:
                    actions = data['robot_joint'][episode_slice]
                    logger.info(f"Using robot_joint fallback - Shape: {actions.shape}")
                    
                    # Re-extract gripper data from robot_joint
                    gripper_actions = ActionExtractor._extract_gripper_from_joints(
                        data, episode_slice, 'robot_joint', actions)
                    
                    # Handle joint dimensions for robot_joint
                    if actions.shape[1] > 6:
                        actions = actions[:, :6]
                        logger.info(f"Extracted 6D joint positions - Shape: {actions.shape}")
                else:
                    raise ValueError("joint_action is all zeros and robot_joint not available. Use EEF mode instead.")
            else:
                # Check if joint_action contains reasonable values
                if np.abs(np.mean(actions)) < 1e-6 and np.std(actions) < 1e-6:
                    logger.warning("joint_action data has very small values - verify this is intended")
        
        return ActionData(actions=actions, gripper_actions=gripper_actions)
    
    @staticmethod
    def _extract_eef_actions(dataset_info: DatasetInfo, episode_slice: slice) -> ActionData:
        """Extract end-effector based actions."""
        data = dataset_info.data
        
        if 'robot_eef_pose' not in data:
            available = [k for k in dataset_info.available_keys if 'eef' in k or 'pose' in k]
            raise KeyError(f"'robot_eef_pose' not found. Available pose keys: {available}")
        
        actions = data['robot_eef_pose'][episode_slice]
        logger.info(f"Using EEF poses - Shape: {actions.shape}")
        logger.info(f"Rotation format: axis-angle, shape: {actions[:,3:6].shape}")
        
        # Extract gripper from EEF data
        gripper_actions = None
        if actions.shape[1] >= 7:
            gripper_actions = actions[:, 6:7]
            logger.info(f"EEF poses include gripper - Shape: {gripper_actions.shape}")
        
        return ActionData(actions=actions, gripper_actions=gripper_actions)
    
    @staticmethod
    def _extract_gripper_from_joints(data: Dict[str, np.ndarray], episode_slice: slice,
                                   joint_source: str, actions: np.ndarray) -> Optional[np.ndarray]:
        """Extract gripper data from joint information."""
        # Try to get gripper from current joint source
        if actions.shape[1] > 6:
            gripper = actions[:, 6:7]
            logger.info(f"Extracted gripper from {joint_source} - Shape: {gripper.shape}")
            return gripper
        
        # Fallback: try to get gripper from robot_joint if using joint_action
        if joint_source == 'joint_action' and 'robot_joint' in data:
            robot_joint_data = data['robot_joint'][episode_slice]
            if robot_joint_data.shape[1] > 6:
                gripper = robot_joint_data[:, 6:7]
                logger.info(f"Using gripper from robot_joint - Shape: {gripper.shape}")
                return gripper
        
        return None


class MovementFilter:
    """Handles filtering of non-moving segments from action sequences."""
    
    @staticmethod
    def filter_actions(dataset_info: DatasetInfo, episode_slice: slice,
                      action_data: ActionData, ctrl_mode: str,
                      movement_threshold: float = 0.001) -> ActionData:
        """Filter out non-moving segments for smoother replay."""
        
        if ctrl_mode == 'joint':
            return MovementFilter._filter_joint_movements(
                dataset_info, episode_slice, action_data)
        else:
            return MovementFilter._filter_pose_movements(
                action_data, movement_threshold)
    
    @staticmethod
    def _filter_joint_movements(dataset_info: DatasetInfo, episode_slice: slice,
                               action_data: ActionData) -> ActionData:
        """Filter joint movements based on joint position differences."""
        data = dataset_info.data
        
        try:
            obs_joints = data['robot_joint'][episode_slice]
            joint_actions = data['joint_action'][episode_slice]
            
            # Compare only first 6 dimensions
            obs_joints_6d = obs_joints[:, :6] if obs_joints.shape[1] > 6 else obs_joints
            joint_actions_6d = joint_actions[:, :6] if joint_actions.shape[1] > 6 else joint_actions
            
            not_moving_mask = are_joints_close(obs_joints_6d, joint_actions_6d)
            moving_mask = ~not_moving_mask
        except KeyError as e:
            logger.warning(f"Could not filter joint movements: {e}")
            moving_mask = np.ones(len(action_data.actions), dtype=bool)
        
        return ActionData(
            actions=action_data.actions[moving_mask],
            gripper_actions=action_data.gripper_actions[moving_mask] 
                           if action_data.gripper_actions is not None else None
        )
    
    @staticmethod
    def _filter_pose_movements(action_data: ActionData, 
                              movement_threshold: float) -> ActionData:
        """Filter pose movements based on position differences."""
        actions = action_data.actions
        pose_diff = np.diff(actions[:, :3], axis=0)
        pose_movement = np.linalg.norm(pose_diff, axis=1)
        
        moving_indices = np.where(pose_movement > movement_threshold)[0] + 1
        moving_mask = np.zeros(len(actions), dtype=bool)
        moving_mask[0] = True  # Always include first pose
        moving_mask[moving_indices] = True
        
        return ActionData(
            actions=actions[moving_mask],
            gripper_actions=action_data.gripper_actions[moving_mask] 
                           if action_data.gripper_actions is not None else None
        )


class RobotReplayer:
    """Handles robot control and action execution."""
    
    def __init__(self, env_config: Dict):
        self.env_config = env_config
    
    def replay_episode(self, action_data: ActionData, ctrl_mode: str, 
                      frequency: int, batch_size: int):
        """Execute action sequence on robot."""
        
        with RealUR5eEnv(**self.env_config) as env:
            logger.info('Environment created, starting replay...')
            
            # Generate timestamps
            timestamps = time.time() + np.arange(len(action_data.actions)) / frequency + 2.0
            start_step = 0
            
            logger.info(f"Replaying {len(action_data.actions)} actions at {frequency} Hz...")
            logger.info("Press Ctrl+C to stop replay")
            
            try:
                while start_step < len(action_data.actions):
                    end_step = min(start_step + batch_size, len(action_data.actions))
                    
                    # Extract batch
                    action_batch = action_data.actions[start_step:end_step]
                    timestamp_batch = timestamps[start_step:end_step]
                    
                    print(f'action_batch: {action_batch}, timestamp_batch: {timestamp_batch}')
                    
                    # Execute actions
                    self._execute_action_batch(
                        env, action_batch, action_data.gripper_actions,
                        timestamp_batch, start_step, end_step, ctrl_mode
                    )
                    
                    logger.info(f'Executed actions {start_step} to {end_step-1} / {len(action_data.actions)}')
                    start_step = end_step
                    
                    # Wait for next batch
                    precise_wait(time.monotonic() + 1.0)
                
                logger.info("Replay completed successfully!")
                
            except KeyboardInterrupt:
                logger.info("Replay interrupted by user")
            except Exception as e:
                logger.error(f"Error during replay: {e}")
                raise
    
    def _execute_action_batch(self, env, action_batch: np.ndarray,
                             gripper_actions: Optional[np.ndarray],
                             timestamp_batch: np.ndarray, start_step: int,
                             end_step: int, ctrl_mode: str):
        """Execute a batch of actions."""
        
        if ctrl_mode == 'joint':
            joint_batch = self._prepare_joint_batch(
                action_batch, gripper_actions, start_step, end_step)
            
            env.exec_actions(
                joint_actions=joint_batch,
                eef_actions=np.zeros((joint_batch.shape[0], 7)),
                timestamps=timestamp_batch,
                mode=ctrl_mode,
            )
        else:  # eef mode
            eef_batch = self._prepare_eef_batch(
                action_batch, gripper_actions, start_step, end_step)
            
            env.exec_actions(
                joint_actions=np.zeros((eef_batch.shape[0], 6)),
                eef_actions=eef_batch,
                timestamps=timestamp_batch,
                mode=ctrl_mode,
            )
    
    def _prepare_joint_batch(self, action_batch: np.ndarray,
                           gripper_actions: Optional[np.ndarray],
                           start_step: int, end_step: int) -> np.ndarray:
        """Prepare joint action batch with optional gripper."""
        joint_batch = action_batch
        
        # Ensure 6D joints
        if joint_batch.shape[1] > 6:
            joint_batch = joint_batch[:, :6]
        
        # Add gripper if available
        if gripper_actions is not None and self.env_config.get('use_gripper', True):
            gripper_batch = gripper_actions[start_step:end_step]
            joint_batch = np.concatenate([joint_batch, gripper_batch], axis=1)
            if start_step == 0:
                logger.info(f"Added gripper to joint actions - Final shape: {joint_batch.shape}")
        
        return joint_batch
    
    def _prepare_eef_batch(self, action_batch: np.ndarray,
                          gripper_actions: Optional[np.ndarray],
                          start_step: int, end_step: int) -> np.ndarray:
        """Prepare EEF action batch with optional gripper."""
        eef_batch = action_batch
        
        # Add gripper if available and not already included
        if (gripper_actions is not None and 
            self.env_config.get('use_gripper', True) and 
            eef_batch.shape[1] < 7):
            gripper_batch = gripper_actions[start_step:end_step]
            eef_batch = np.concatenate([eef_batch, gripper_batch], axis=1)
            if start_step == 0:
                logger.info(f"Added gripper to EEF actions - Final shape: {eef_batch.shape}")
        
        return eef_batch


def get_episode_slice(episode_num: int, episode_ends: np.ndarray) -> slice:
    """Get slice for specific episode."""
    if episode_num >= len(episode_ends):
        raise ValueError(f"Episode {episode_num} not found. Available: 0-{len(episode_ends)-1}")
    
    episode_start = 0 if episode_num == 0 else episode_ends[episode_num-1]
    episode_end = episode_ends[episode_num]
    
    logger.info(f"Replaying episode {episode_num}: timesteps {episode_start} to {episode_end-1}")
    return slice(episode_start, episode_end)


def create_env_config(args) -> Dict:
    """Create environment configuration from arguments."""
    return {
        'output_dir': 'tmp_replay',
        'ctrl_mode': args.mode,
        'speed_slider_value': args.speed,
        'single_arm_type': args.arm,
        'use_gripper': not args.no_gripper,
        'robot_left_ip': args.robot_left_ip,
        'robot_right_ip': args.robot_right_ip,
        'tactile_sensors': None
    }


def replay_episode_pipeline(args):
    """Main pipeline for episode replay."""
    # Load dataset
    dataset_info = DatasetLoader.load(args.dataset)
    
    # Get episode slice
    episode_slice = get_episode_slice(args.episode, dataset_info.episode_ends)
    
    # Extract actions
    action_data = ActionExtractor.extract(
        dataset_info, episode_slice, args.mode, args.joint_source)
    
    # # Filter movements if requested
    # if not args.no_filter:
    #     action_data = MovementFilter.filter_actions(
    #         dataset_info, episode_slice, action_data, args.mode)
    #     logger.info(f"Filtered actions - Original: {len(ActionExtractor.extract(dataset_info, episode_slice, args.mode, args.joint_source).actions)}, Moving: {len(action_data.actions)}")
    
    # Check for tactile data
    if 'left_tactile' in dataset_info.data and 'right_tactile' in dataset_info.data:
        left_tactile = dataset_info.data['left_tactile'][episode_slice]
        right_tactile = dataset_info.data['right_tactile'][episode_slice]
        logger.info(f"Tactile data found - Left: {left_tactile.shape}, Right: {right_tactile.shape}")
    
    # Create and run replayer
    env_config = create_env_config(args)
    replayer = RobotReplayer(env_config)
    replayer.replay_episode(action_data, args.mode, args.frequency, args.batch_size)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Replay zarr dataset on UR5e robot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/puzzle_expert_30.zarr
  %(prog)s data/puzzle_expert_30.zarr -m joint -j robot_joint
  %(prog)s demo_data_raw/puzzle_expert_50_raw/replay_buffer.zarr -s 0.1
        """
    )
    
    # Required arguments
    parser.add_argument('dataset', help='Path to zarr dataset')
    
    # Control arguments
    parser.add_argument('-e', '--episode', type=int, default=0,
                       help='Episode number to replay (default: %(default)s)')
    parser.add_argument('-m', '--mode', choices=['joint', 'eef'], default='eef',
                       help='Control mode (default: %(default)s)')
    parser.add_argument('-j', '--joint-source', choices=['robot_joint', 'joint_action'], 
                       default='robot_joint',
                       help='Joint data source (default: %(default)s)')
    parser.add_argument('-s', '--speed', type=float, default=0.1,
                       help='Speed scaling factor 0.0-1.0 (default: %(default)s)')
    
    # Execution parameters
    parser.add_argument('-f', '--frequency', type=int, default=3,
                       help='Control frequency in Hz (default: %(default)s)')
    parser.add_argument('-b', '--batch-size', type=int, default=3,
                       help='Actions per batch (default: %(default)s)')
    parser.add_argument('--no-filter', action='store_true', default=True,
                       help='Disable movement filtering')
    
    # Robot configuration
    parser.add_argument('--robot-left-ip', default='192.168.0.3',
                       help='Left robot IP (default: %(default)s)')
    parser.add_argument('--robot-right-ip', default='192.168.0.2',
                       help='Right robot IP (default: %(default)s)')
    parser.add_argument('--arm', choices=['left', 'right'], default='right',
                       help='Arm for single-arm control (default: %(default)s)')
    parser.add_argument('--no-gripper', action='store_true',
                       help='Disable gripper control')
    
    return parser


def print_configuration(args):
    """Print configuration summary."""
    print("=== Dataset Replay Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Episode: {args.episode}")
    print(f"Control mode: {args.mode}")
    if args.mode == 'joint':
        print(f"Joint source: {args.joint_source}")
    print(f"Speed: {args.speed}")
    print(f"Frequency: {args.frequency} Hz")
    print(f"Batch size: {args.batch_size}")
    print(f"Filter movement: {not args.no_filter}")
    print(f"Arm: {args.arm}")
    print(f"Gripper: {not args.no_gripper}")
    print("=" * 40)


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found at {args.dataset}")
        sys.exit(1)
    
    print_configuration(args)
    
    # Safety confirmation
    try:
        response = input("Is the robot workspace clear and emergency stop ready? (y/N): ")
        if response.lower() != 'y':
            logger.info("Replay cancelled for safety")
            sys.exit(0)
    except (EOFError, KeyboardInterrupt):
        logger.info("Replay cancelled")
        sys.exit(0)
    
    # Run replay pipeline
    try:
        replay_episode_pipeline(args)
    except Exception as e:
        logger.error(f"Replay failed: {e}")
        raise


if __name__ == '__main__':
    main()
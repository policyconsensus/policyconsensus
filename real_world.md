# Real-World Robotic Manipulation Guide

This guide covers the complete workflow for real-world robotic manipulation with UR5e robots, from data collection to policy evaluation.

## Hardware Requirements

Before starting, ensure you have the following hardware:

**Required:**

- **UR5e Robot Arm** - Universal Robots collaborative robot
- **Gripper** - Robotiq or compatible parallel jaw gripper
- **Cameras** - Intel RealSense D415 cameras 
- **SpaceMouse** - 3D navigation device for teleoperation

**Optional (for multi-modal manipulation):**

- **Tactile Sensors** - 2x FlexiTac sensors (12×32 sensing units per finger)

**Network:**

- Ethernet connection to robot(s) (default IPs: 192.168.0.2)

## Software Prerequisites

**Environment Setup:**

Before starting, set up the environment for real robot hardware:

```bash
./setup.sh --real
```

This creates the `policy-consensus` conda environment with real robot dependencies (RealSense cameras, UR-RTDE, etc.).

**After setup:**

```bash
conda activate policy-consensus
source ~/.bashrc  # If first time
```

For camera calibration, see: <https://github.com/Tool-as-Interface/Tool_as_Interface>

## Pre-Collected Datasets

Download pre-collected demonstration dataset for real-world tasks from Hugging Face:

**Available Dataset:**

- **Puzzle Insertion** (50 demos): [Download from Hugging Face](https://huggingface.co/datasets/haonan-chen/policy-consensus/resolve/main/puzzle_expert_50.zarr.tar.gz)

**Usage:**

```bash
# Download and extract dataset
wget https://huggingface.co/datasets/haonan-chen/policy-consensus/resolve/main/puzzle_expert_50.zarr.tar.gz -O puzzle_expert_50.zarr.tar.gz
mkdir -p data
tar -xzf puzzle_expert_50.zarr.tar.gz -C data/

# Verify dataset
python scripts/inspect_and_replay_dataset.py data/puzzle_expert_50.zarr --inspect-only
```

## Quick Start & Workflow Overview

**New to the system?** Follow this streamlined workflow:

1. **Collect demonstrations**: `python scripts/demo_real_ur5e.py`
2. **Process data**: `python scripts/convert_demo_data.py -d data/raw_demos`
3. **Inspect dataset**: `python scripts/inspect_and_replay_dataset.py data/processed.zarr`
4. **Train policy**: See [Policy Training](#policy-training) section below
5. **Evaluate policy**: `python modular_policy/workspace/eval_policy_real.py -i checkpoint.ckpt -o results/`

## Table of Contents

1. [Quick Start & Workflow Overview](#quick-start--workflow-overview)
2. [Data Collection](#data-collection)
3. [Data Processing](#data-processing)
4. [Dataset Inspection](#dataset-inspection)
5. [Dataset Replay](#dataset-replay)
6. [Policy Training](#policy-training)
7. [Policy Evaluation](#policy-evaluation)
8. [Adding New Tasks](#adding-new-tasks)

---

## Data Collection

Collect real-world demonstrations using the UR5e robot setup.

### Basic Collection

```bash
python scripts/demo_real_ur5e.py
```

### Key Features

- **Interactive control**: Use SpaceMouse and keyboard for demonstration
- **Multi-camera recording**: Captures multiple camera views
- **Tactile feedback**: Records tactile sensor data during manipulation

### Safety Notes

- Ensure workspace is clear of obstacles
- Keep emergency stop button accessible
- Start with slow movements to test setup

---

## Data Processing

```bash
python scripts/convert_demo_data.py -d /path/to/raw/data
```

**Common options**:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `-d, --data_dir` | **Required** - Raw data directory | - |
| `-s, --save_path` | Output zarr file path | `{data_dir}.zarr` |
| `-v, --vis_size` | Image resize format "HxW" | `"96x128"` |
| `--generate_video` | Create inspection videos | False |
| `-o, --keys` | Specific observation keys to include in the processed dataset. If not specified, all available keys are included. | All available |

**Note**: Large datasets may take 3-4 hours to process.

---

## Dataset Inspection

```bash
# Full inspection + video generation
python scripts/inspect_and_replay_dataset.py data/puzzle_expert_50.zarr

# Fast inspection only (recommended for large datasets)
python scripts/inspect_and_replay_dataset.py data/puzzle_expert_50_raw/replay_buffer.zarr --inspect-only --no-samples
```

**Modes**: Default runs both inspection and video generation. Use `--inspect-only` for fast checks, `--video-only` to skip inspection, or `--episodes` to select specific episodes.

---

## Dataset Replay

Replay recorded demonstrations on real robots using the dedicated replay script.

### Quick Start

```bash
# Basic end-effector replay
python scripts/replay_dataset.py data/puzzle_expert_50.zarr

# Slow and safe replay
python scripts/replay_dataset.py data/puzzle_expert_50.zarr -s 0.1 -f 5
```

**Common options**: `-e` (episode), `-s` (speed 0-1), `-m` (mode: `eef` or `joint`), `-j` (joint source: `robot_joint` or `joint_action`). Run with `--help` for all options.

### Joint Data Sources

**For most users**: Use default settings (EEF mode or joint mode with `robot_joint`).

**Joint mode options** (`-m joint`):

- `robot_joint` (default, recommended): Uses actual recorded joint positions - always safe
- `joint_action` (avoid): Uses joint commands - often zeros for EEF-collected data, can be dangerous

**Note**: The script automatically protects against problematic data and switches to safe defaults.

---

## Policy Training

Train policies for real-world robotic manipulation tasks.

### Basic Training

**Ours - Compositional (RGB + Tactile):**

```bash
python scripts/manip_policy.py \
  --config-name="train_dp_unets_spec_rgb_tactile" \
  task=real/puzzle_expert \
  task.dataset.zarr_path=data/puzzle_expert_50.zarr \
  training.num_epochs=1001 \
  policy.obs_encoders.0.model_library.RobomimicRgbEncoder.crop_shape='[91, 121]' \
  policy.obs_encoders.1.model_library.RobomimicTactileEncoder.crop_shape='[15, 30]'
```

**Baselines:**

```bash
# Feature Concatenation (RGB + Tactile)
python scripts/manip_policy.py \
  --config-name="train_dp_unet_rgb_tactile" \
  task=real/puzzle_expert \
  task.dataset.zarr_path=data/puzzle_expert_50.zarr \
  training.num_epochs=1001

# RGB Only
python scripts/manip_policy.py \
  --config-name="train_dp_unet_rgb" \
  task=real/puzzle_expert \
  task.dataset.zarr_path=data/puzzle_expert_50.zarr \
  task.dataset.obs_keys="['camera_0_color','camera_1_color','robot_joint']"
```

See `modular_policy/config/` for all available configurations.

---

## Policy Evaluation

```bash
python modular_policy/workspace/eval_policy_real.py \
    -i output/checkpoints/policy.ckpt \
    -o output/eval_results/
```

**Common options**: `-i` (checkpoint path), `-o` (output directory), `--frequency` (control Hz), `--use_tactile` (enable tactile). Run with `--help` for all options.

### 3. Safety & Controls

**Keyboard controls:**

- **'C'**: Start policy | **'S'**: Stop policy (return to manual control)
- **'H'**: Home position | **'Q'**: Quit

**Manual control**: Use SpaceMouse for end-effector positioning. Keep emergency stop accessible at all times.

### 4. Critical Safety Protocol

**CRITICAL SAFETY REQUIREMENTS:** Adhere to these protocols to ensure safe operation:

1. **Emergency Stop**: Always keep the hardware emergency stop button within immediate reach and be prepared to use it.
2. **Speed Control**: Begin all operations with low speeds and gradually increase only after verifying safe behavior.
3. **Clear Workspace**: Ensure the robot's workspace is completely clear of obstacles, personnel, and any items not part of the task.
4. **Continuous Monitoring**: Closely observe the robot's movements and behavior throughout policy execution.
5. **Quick Response**: Be ready to press 'S' (stop policy) or the hardware emergency stop at any sign of unexpected behavior.

---

## Adding New Tasks

To add a new real-world task:

1. **Create config**: Copy `modular_policy/config/task/real/puzzle_expert.yaml` and modify for your task
2. **Collect demos**: `python scripts/demo_real_ur5e.py -o data/my_task_raw`
3. **Process data**: `python scripts/convert_demo_data.py -d data/my_task_raw -s data/my_task.zarr`
4. **Inspect data**: `python scripts/inspect_and_replay_dataset.py data/my_task.zarr`
5. **Train policy**: `python scripts/manip_policy.py --config-name=train_dp_unet_rgb_tactile task=real/my_task task.dataset.zarr_path=data/my_task.zarr`

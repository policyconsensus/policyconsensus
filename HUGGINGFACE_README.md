---
license: mit
task_categories:
- robotics
tags:
- manipulation
- multi-modal
- diffusion-policy
- tactile
- real-world
---

# Multi-Modal Policy Consensus - Datasets

This repository contains demonstration datasets for the paper **"Multi-Modal Manipulation via Multi-Modal Policy Consensus"**.

📄 [Paper](https://arxiv.org/abs/2509.23468) | 💻 [Code](https://github.com/your-repo-url) | 🌐 [Project Page](https://your-website-url.github.io)

## Datasets

### Simulation (RLBench MT4)
**File:** `mt4_expert_200.zarr.tar.gz` 

- **Tasks:** 4 multi-task manipulation benchmarks (open_box, open_drawer, take_umbrella_out_of_umbrella_stand, toilet_seat_up)
- **Episodes:** 50 demonstrations per task (200 total)
- **Modalities:** RGB images, point clouds, DINO features, proprioception
- **Environment:** RLBench simulation

### Real-World (UR5e Robot)
**File:** `puzzle_expert_50.zarr.tar.gz` 
- **Task:** Puzzle insertion
- **Episodes:** 50 demonstrations
- **Modalities:** RGB images (2 cameras), tactile sensing (FlexiTac), proprioception
- **Robot:** UR5e with Robotiq gripper

## Download & Usage

```bash
# Download simulation dataset
wget https://huggingface.co/datasets/haonan-chen/policy-consensus/resolve/main/mt4_expert_200.zarr.tar.gz
mkdir -p data/rlbench
tar -xzf mt4_expert_200.zarr.tar.gz -C data/rlbench/

# Download real-world dataset
wget https://huggingface.co/datasets/haonan-chen/policy-consensus/resolve/main/puzzle_expert_50.zarr.tar.gz
mkdir -p data
tar -xzf puzzle_expert_50.zarr.tar.gz -C data/

# Verify datasets
python scripts/inspect_and_replay_dataset.py data/rlbench/mt4_expert_200.zarr --inspect-only
python scripts/inspect_and_replay_dataset.py data/puzzle_expert_50.zarr --inspect-only
```

For complete setup, training, and evaluation instructions, see the [main repository](https://github.com/your-repo-url).

## Dataset Format

All datasets are stored in Zarr format with the following structure:

```
dataset.zarr/
├── data/
│   ├── action          # Robot actions (end-effector poses, gripper)
│   ├── camera_*_color  # RGB images from cameras
│   ├── tactile_*       # Tactile sensor readings (real-world only)
│   └── robot_joint     # Joint positions and velocities
└── meta/
    └── episode_ends    # Episode boundary indices
```

## Citation

If you use these datasets in your research, please cite:

```bibtex
@misc{chen2025multimodalmanipulationmultimodalpolicy,
      title={Multi-Modal Manipulation via Multi-Modal Policy Consensus},
      author={Haonan Chen and Jiaming Xu and Hongyu Chen and Kaiwen Hong and Binghao Huang and Chaoqi Liu and Jiayuan Mao and Yunzhu Li and Yilun Du and Katherine Driggs-Campbell},
      year={2025},
      eprint={2509.23468},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.23468},
}
```

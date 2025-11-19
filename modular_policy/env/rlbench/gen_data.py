if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import pathlib
import click
import numpy as np
import copy
import zarr
import multiprocessing as mp

from modular_policy.common.replay_buffer import ReplayBuffer
from modular_policy.common.input_util import wait_user_input
from modular_policy.env.rlbench.env import RlbenchEnv
from modular_policy.env.rlbench.factory import get_subtasks

from typing import List, Dict


def generate_one_episode_data(env: RlbenchEnv):
    demo = env.task_env.get_demos(1, live_demos=True, max_attempts=1)
    demo = demo[0]      # only one episode
    this_total_count = len(demo)
    this_data_collected = {'action': []}

    for obs in demo:
        obs_dict = env._extact_obs(obs)
        for k, v in obs_dict.items():
            if k not in this_data_collected:
                this_data_collected[k] = []
            this_data_collected[k].append(v)
        # absolute joint position control
        action = np.append(obs.joint_positions, obs.gripper_joint_positions)
        this_data_collected['action'].append(action)

    return this_total_count, this_data_collected


def worker(
    worker_id: int,
    task_names: List[str],
    task_idices: List[int],
    sensors: List[str],
    data_dtype: Dict,
    save_subtasks: bool,
) -> List[ReplayBuffer]:
    seed = os.getpid()
    env = RlbenchEnv(
        enable_depth=True, 
        num_points=512, 
        enable_dino=True,
        pca_reduction=None,
        camera_names=sensors, 
        seed=seed
    )
    replay_buffers = [
        ReplayBuffer.create_empty_zarr() for _ in range(
            (len(task_names) + 1) if save_subtasks else 1
        )
    ]

    idx = 0
    while idx < len(task_idices):
        task_idx = task_idices[idx]
        task_name = task_names[task_idx]
        env.set_task(task_name)
        
        try:
            this_total_count, this_data_collected = generate_one_episode_data(env)
        except RuntimeError as e:
            if str(e) == 'Could not collect demos. Maybe a problem with the task?':
                # skip this episode
                print(f"Worker {worker_id} - [{idx+1}/{len(task_idices)}]: task: {env.task_name}, failed")
                continue
            else:
                raise e

        idx += 1
        # dtype conversion
        for k, v in this_data_collected.items():
            this_data_collected[k] = np.array(v, dtype=data_dtype.get(k, 'float32'))
        # copy trial data to total data collected
        replay_buffers[-1].add_episode(copy.deepcopy(this_data_collected))
        # copy trial data to subtask data collected if needed
        if save_subtasks:
            replay_buffers[task_idx].add_episode(copy.deepcopy(this_data_collected))
        print(
            f"Worker {worker_id} - [{idx}/{len(task_idices)}]: "
            f"task: {env.task_name}, step: {this_total_count}"
        )

    env.close()
    return replay_buffers


def multiprocess_core(
    num_workers: int,
    task_name: str,
    sensors: str,
    save_subtasks: bool,
    num_tasks: int,
    num_eps_per_task: int
) -> List[ReplayBuffer]:
    replay_buffers = [
        ReplayBuffer.create_empty_zarr() for _ in range(
            (num_tasks + 1) if save_subtasks else 1
        )
    ]
    data_dtype = {
        **{f"{sensor}_rgb": 'uint8' for sensor in sensors},
        **{f"{sensor}_depth": 'float32' for sensor in sensors},
    }

    # collect data
    task_indices = list(range(num_tasks)) * num_eps_per_task
    with mp.Pool(num_workers) as pool:
        args = [
            (
                i,
                get_subtasks(task_name),
                task_indices[i::num_workers],
                sensors,
                data_dtype,
                save_subtasks
            ) for i in range(num_workers)
        ]
        results = pool.starmap(worker, args)
        pool.close()
        pool.join()

    # merge data
    for buff_idx in range(len(replay_buffers)):
        for worker_idx in range(num_workers):
            for eps_idx in range(results[worker_idx][buff_idx].n_episodes):
                replay_buffers[buff_idx].add_episode(
                    results[worker_idx][buff_idx].get_episode(eps_idx)
                )
    return replay_buffers


def uniprocess_core(
    task_name: str,
    sensors: str,
    save_subtasks: bool,
    num_tasks: int,
    num_eps_per_task: int
) -> List[ReplayBuffer]:
    replay_buffers = [
        ReplayBuffer.create_empty_zarr() for _ in range(
            (num_tasks + 1) if save_subtasks else 1
        )
    ]
    data_dtype = {
        **{f"{sensor}_rgb": 'uint8' for sensor in sensors},
        **{f"{sensor}_depth": 'float32' for sensor in sensors},
    }

    env = RlbenchEnv(
        enable_depth=True,
        num_points=512, 
        enable_dino=True,
        pca_reduction=None,
        camera_names=sensors, 
    )

    for task_idx, task_name in enumerate(get_subtasks(task_name)):
        env.set_task(task_name)

        episode_idx = 0
        while episode_idx < num_eps_per_task:
            try:
                this_total_count, this_data_collected = generate_one_episode_data(env)
            except RuntimeError as e:
                if str(e) == 'Could not collect demos. Maybe a problem with the task?':
                    # skip this episode
                    print(f"Episode {episode_idx + num_eps_per_task * task_idx + 1} failed. Skip")
                    continue
                else:
                    raise e
                
            episode_idx += 1
            # dtype conversion
            for k, v in this_data_collected.items():
                this_data_collected[k] = np.array(v, dtype=data_dtype.get(k, 'float32'))
            # copy trial data to total data collected
            replay_buffers[-1].add_episode(copy.deepcopy(this_data_collected))
            # copy trial data to subtask data collected if needed
            if save_subtasks:
                replay_buffers[task_idx].add_episode(copy.deepcopy(this_data_collected))
            print(
                f"Episode {episode_idx + num_eps_per_task * task_idx}, "
                f"task: {env.task_name}, step: {this_total_count}"
            )

    env.close()
    return replay_buffers


@click.command()
@click.option('-t', '--task_name', type=str, required=True)
@click.option('-d', '--root_data_dir', type=str, default='data/rlbench')
@click.option('-s', '--save_dir', type=str, default=None)
@click.option('-c', '--num_episodes', type=int, default=10)
@click.option('-o', '--sensors', multiple=True, type=str, default=(
    'left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'))
@click.option('-w', '--num_workers', type=int, default=1)
@click.option('--save_subtasks', is_flag=True)
def gen_rlbench_data(
    task_name: str,
    root_data_dir: str,
    save_dir: str,
    num_episodes: int,
    sensors: str,
    num_workers: int,
    save_subtasks: bool
):  
    assert num_workers > 0

    if save_dir is None:
        save_dir = os.path.join(root_data_dir, f'{task_name}_expert_{num_episodes}.zarr')

    if os.path.exists(save_dir):
        keypress = wait_user_input(
            valid_input=lambda key: key in ['', 'y', 'n'],
            prompt=f"{save_dir} already exists. Overwrite? [y/`n`]: ",
            default='n'
        )
        if keypress == 'n':
            print("Abort")
            return
        else:
            os.system(f"rm -rf {save_dir}")
    pathlib.Path(save_dir).mkdir(parents=True)
    save_dir = [save_dir,]

    # distribute episodes across different environments
    num_tasks = len(get_subtasks(task_name))
    assert num_episodes % num_tasks == 0
    num_eps_per_task = num_episodes // num_tasks
    assert not save_subtasks or num_tasks > 1

    # create subtask directories
    if save_subtasks:
        for sname in get_subtasks(task_name):
            sdir = os.path.join(root_data_dir, f'{sname}_expert_{num_eps_per_task}.zarr')
            if os.path.exists(sdir):
                keypress = wait_user_input(
                    valid_input=lambda key: key in ['', 'y', 'n'],
                    prompt=f"{sdir} already exists. Overwrite? [y/`n`]: ",
                    default='n'
                )
                if keypress == 'n':
                    print("Abort")
                    return
                else:
                    os.system(f"rm -rf {sdir}")
            os.mkdir(sdir)
            save_dir.append(sdir)
    save_dir = save_dir[1:] + save_dir[:1]

    # collect data
    if num_workers == 1:
        replay_buffers = uniprocess_core(
            task_name,
            sensors,
            save_subtasks,
            num_tasks,
            num_eps_per_task
        )
    elif num_workers > 1:
        mp.set_start_method('spawn', force=True)
        replay_buffers = multiprocess_core(
            num_workers,
            task_name,
            sensors,
            save_subtasks,
            num_tasks,
            num_eps_per_task
        )

    # report
    for buff, sdir in zip(replay_buffers, save_dir):
        print('-' * 50)
        print(f"{sdir}: \n{buff}")

    # save data collected
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=1)
    for i, buff in enumerate(replay_buffers):
        buff.save_to_path(save_dir[i], compressors=compressor)


if __name__ == '__main__':
    gen_rlbench_data()
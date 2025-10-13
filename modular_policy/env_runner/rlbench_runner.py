import os
import warnings
import wandb
import numpy as np
import torch
import tqdm
import math
import pathlib
import dill
import wandb.sdk.data_types.video as wandb_video
from rlbench.action_modes.action_mode import ActionMode

from modular_policy.gymnasium_util.multistep_wrapper import MultiStepWrapper
from modular_policy.gymnasium_util.video_recording_wrapper import (
    AsyncVecRlbenchVideoRecordingWrapper, VideoRecordingWrapper, VideoRecorder)
from modular_policy.gymnasium_util.async_vector_env import AsyncVectorEnv
from modular_policy.env.rlbench.env import RlbenchEnv
from modular_policy.env.rlbench.factory import get_subtasks
from modular_policy.env_runner.base_runner import BaseRunner, obs_dropout
from modular_policy.policy.base_policy import BasePolicy
from modular_policy.common.pytorch_util import dict_apply

from typing import Optional, List



def merge_obs_dicts(obs_dicts):
    # X obs_dicts, each Dict[str, np.ndarray]
    # return Dict[str, np.ndarray]
    return {
        key: np.stack([obs_dict[key] for obs_dict in obs_dicts])
        for key in obs_dicts[0].keys()
    }


class RlbenchParallelRunner(BaseRunner):
    """
    The async vec rlbench env must use spawn start method. Which leads
    to bug in video recording. The resulting videos may contain more than
    one episode/trial, caused by unflashed video buffer. Currently, I
    don't know how to fix this. When exact video is needed, use the
    non-parallel runner. At the cost of speed. Set environment variable
    `PARALLEL_RLBENCH=0` to use the non-parallel runner.
    """

    import multiprocessing as mp
    # mp.set_start_method('spawn', force=True)    # NOTE: spawn is required
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    def __init__(self,
        output_dir,
        task_name: str,
        n_test: int = 22,
        n_test_vis: int = 6,
        test_start_seed: int = 1000,
        n_obs_steps: int = 8,
        n_action_steps: int = 8,
        fps: int = 10,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_parallel_envs: Optional[int] = None,
        # other rlbench env args
        image_size: int = 128,
        num_points: int = 512,
        enable_depth: bool = False,
        enable_dino: bool = False,
        image_scale: float = 255.0,
        seed: Optional[int] = None,
        camera_names: List[str] = ['left_shoulder','right_shoulder','overhead','wrist','front'],
        robot_state_ports: List[str] = [
            'joint_positions', 'joint_velocities', 'joint_forces',
            'gripper_open', 'gripper_pose', 'gripper_joint_positions',
            'gripper_touch_forces'
        ],
        max_episode_steps: int = 250,
        action_mode: Optional[ActionMode] = None,
        obs_to_drop: List[str] = [],
        zarr_path: Optional[str] = None,
    ):
        super().__init__(output_dir)

        if n_parallel_envs is None:
            n_parallel_envs = n_test

        # get subtasks and distribute across envs
        subtask_names = get_subtasks(task_name)
        env_task_names = [
            subtask_names[i % len(subtask_names)] 
            for i in range(n_test)
        ]

        # setup env
        env_seeds = []
        env_fns = []
        env_init_fn_dills = []
        for i in range(n_test):
            this_task_name = env_task_names[i]
            seed = test_start_seed + i
            env_seeds.append(seed)
            enable_render = i < n_test_vis

            if i < n_parallel_envs:
                def env_fn(task_name=this_task_name, seed=seed):
                    return MultiStepWrapper(
                        AsyncVecRlbenchVideoRecordingWrapper(
                            RlbenchEnv(
                                task_name=task_name,
                                image_size=image_size,
                                num_points=512, # num_points,
                                enable_depth=enable_depth,
                                enable_dino=enable_dino,
                                process_dino=True,
                                image_scale=image_scale,
                                seed=seed,
                                camera_names=camera_names,
                                robot_state_ports=robot_state_ports,
                                max_episode_steps=max_episode_steps,
                                action_mode=action_mode,
                                zarr_path=zarr_path,
                            ),
                            video_recoder=VideoRecorder.create_h264(
                                fps=fps,
                                codec='h264',
                                input_pix_fmt='rgb24',
                                crf=crf,
                                thread_type='FRAME',
                                thread_count=1
                            ),
                            file_path=None,
                            steps_per_render=1
                        ),
                        n_obs_steps=n_obs_steps,
                        n_action_steps=n_action_steps,
                        max_episode_steps=max_episode_steps,
                        reward_agg_method='sum'
                    )
                env_fns.append(env_fn)

            def init_fn(env, task_name=this_task_name, seed=seed, enable_render=enable_render):
                if not env.env.env.task_name.startswith(task_name):
                    raise RuntimeError(
                        f"Env task name: {env.env.env.task_name} does not "
                        f"match expected task name: {task_name}."
                    )
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        f'media/{task_name}', wandb_video.util.generate_id() + '.mp4'
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                env.reset(seed=seed)
            env_init_fn_dills.append(dill.dumps(init_fn))

        assert len(env_fns) == n_parallel_envs
        assert len(env_init_fn_dills) == n_test

        env = AsyncVectorEnv(env_fns)

        # attr assignment
        self.task_name = task_name
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_task_names = env_task_names
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_to_drop = obs_to_drop


    @torch.inference_mode()
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_success = [False] * n_inits
        all_avg_weights = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            self.env.call_each(
                'run_dill_function', 
                args_list=[(x,) for x in this_init_fns]
            )

            # start rollout
            obs, _ = self.env.reset()
            policy.reset()
            chunk_episode_weights = [[] for _ in range(this_n_active_envs)]

            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Eval {policy_name} in RLBench::{self.task_name} {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec
            )

            done = False
            while not done and pbar.n < pbar.total:
                # create obs dict
                obs_dict = dict_apply(
                    merge_obs_dicts(obs),
                    lambda x: torch.from_numpy(x).to(device=device, dtype=dtype)
                )
                obs_dict = obs_dropout(obs_dict, self.obs_to_drop)

                # run policy
                with torch.inference_mode():
                    policy_output = policy.predict_action({
                        port: obs_dict[port] 
                        for port in policy.get_observation_ports()
                    })
                    action = policy_output['action'].detach().cpu().numpy()
                    raw_weights = policy_output.get('weights', None)
                    if raw_weights is not None:
                        raw_weights = raw_weights.detach().cpu()
                        for env_i in range(this_n_active_envs):
                            chunk_episode_weights[env_i].append(raw_weights[env_i])

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("NaN of Inf action")

                # step env
                obs, reward, done, _, _ = self.env.step(action) # NOTE: reward=1 if success
                done = np.logical_or(
                    done[this_local_slice],
                    all_success[this_global_slice][this_local_slice]
                )
                done = np.all(done[this_local_slice])

                all_success[this_global_slice] = np.logical_or(
                    all_success[this_global_slice],
                    [r >= 1 for r in reward[this_local_slice]]
                )

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()
            
            # Compute average weights per env in this chunk
            avg_weights = []
            for weights_list in chunk_episode_weights:
                if weights_list:
                    avg_w = np.mean(np.stack(weights_list, axis=0), axis=0)
                else:
                    avg_w = None
                avg_weights.append(avg_w)
            all_avg_weights[this_global_slice] = avg_weights

            # collect data for this round
            all_video_paths[this_global_slice] = self.env.render()[this_local_slice]

        # clear out video buffer
        _ = self.env.reset()
        _ = self.env.render()   # NOTE: hack, rlbench env buffer clean up in render()

        # log
        log_data = dict()

        for task_name in set(self.env_task_names):
            task_indices = [i for i, name in enumerate(self.env_task_names) if name == task_name]

            task_success = [all_success[i] for i in task_indices]
            task_weights = [all_avg_weights[i] for i in task_indices if all_avg_weights[i] is not None]

            log_data[f"{task_name}/mean_success_rate"] = np.mean(task_success)

            if task_weights:
                mean_weight = np.mean(np.stack(task_weights, axis=0), axis=0)  # shape: [n_components]
                softmaxed = np.exp(mean_weight) / np.exp(mean_weight).sum()
                log_data.update({
                    f"{task_name}/avg_raw_weight_{j}": w
                    for j, w in enumerate(mean_weight)
                })
                log_data.update({
                    f"{task_name}/avg_softmax_weight_{j}": w
                    for j, w in enumerate(softmaxed)
                })

        for i in range(n_inits):
            seed = self.env_seeds[i]
            task_name = self.env_task_names[i]

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                video = wandb.Video(video_path)
                log_data[f"{task_name}/video_{seed}"] = video
            
        # log aggregate metrics
        log_data['mean_success_rate'] = np.mean(all_success)
        
        return log_data
    

    def close(self):
        self.env.close()



class RlbenchNonParallelRunner(BaseRunner):
    def __init__(self,
        output_dir,
        task_name: str,
        n_test: int = 22,
        n_test_vis: int = 6,
        test_start_seed: int = 1000,
        n_obs_steps: int = 8,
        n_action_steps: int = 8,
        fps: int = 10,
        crf: int = 22,
        tqdm_interval_sec: float = 5.0,
        n_parallel_envs: Optional[int] = None,
        # other rlbench env args
        image_size: int = 128,
        num_points: int = 512,
        enable_depth: bool = True,
        enable_dino: bool = False,
        image_scale: float = 255.0,
        seed: Optional[int] = None,
        camera_names: List[str] = ['left_shoulder','right_shoulder','overhead','wrist','front'],
        robot_state_ports: List[str] = [
            'joint_positions', 'joint_velocities', 'joint_forces',
            'gripper_open', 'gripper_pose', 'gripper_joint_positions',
            'gripper_touch_forces'
        ],
        max_episode_steps: int = 250,
        action_mode: Optional[ActionMode] = None,
        obs_to_drop: List[str] = [],
        zarr_path: Optional[str] = None,
    ):
        super().__init__(output_dir)

        if n_parallel_envs is not None and n_parallel_envs > 1:
            warnings.warn(
                "n_parallel_envs > 1 is not supported in non-parallel mode. "
                "Setting n_parallel_envs to 1."
            )
            n_parallel_envs = 1

        # get subtasks and distribute across envs
        subtask_names = get_subtasks(task_name)
        env_task_names = [
            subtask_names[i % len(subtask_names)] 
            for i in range(n_test)
        ]

        # setup env
        env_seeds = []
        env_init_fn_dills = []
        for i in range(n_test):
            this_task_name = env_task_names[i]
            seed = test_start_seed + i
            env_seeds.append(seed)
            enable_render = i < n_test_vis

            def init_fn(env, task_name=this_task_name, seed=seed, enable_render=enable_render):
                env.env.env.set_task(task_name, seed=seed)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        f'media/{task_name}', wandb_video.util.generate_id() + '.mp4'
                    )
                    filename.parent.mkdir(parents=True, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename
                env.reset(seed=seed)
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = MultiStepWrapper(
            VideoRecordingWrapper(
                RlbenchEnv(
                    task_name=this_task_name,
                    image_size=image_size,
                    num_points=num_points,
                    enable_depth=enable_depth,
                    enable_dino=enable_dino,
                    process_dino=True,
                    image_scale=image_scale,
                    seed=seed,
                    camera_names=camera_names,
                    robot_state_ports=robot_state_ports,
                    max_episode_steps=max_episode_steps,
                    action_mode=action_mode,
                    zarr_path=zarr_path,
                ),
                video_recoder=VideoRecorder.create_h264(
                    fps=fps,
                    codec='h264',
                    input_pix_fmt='rgb24',
                    crf=crf,
                    thread_type='FRAME',
                    thread_count=1
                ),
                file_path=None,
                steps_per_render=1
            ),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_episode_steps,
            reward_agg_method='sum'
        )

        assert len(env_init_fn_dills) == n_test

        # attr assignment
        self.task_name = task_name
        self.env = env
        self.env_seeds = env_seeds
        self.env_task_names = env_task_names
        self.env_init_fn_dills = env_init_fn_dills
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_episode_steps = max_episode_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_to_drop = obs_to_drop

    
    @torch.inference_mode()
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        policy_name = policy.get_policy_name()

        # plan for rollout
        n_inits = len(self.env_init_fn_dills)

        # allocate data
        all_video_paths = [None] * n_inits
        all_success = [False] * n_inits

        for test_i in range(n_inits):
            init_fn = dill.loads(self.env_init_fn_dills[test_i])
            init_fn(self.env)

            # start rollout
            # obs, _ = self.env.reset(seed=42)    # NOTE: seeding happens here, init_fn has bug
            obs, _ = self.env.reset()
            policy.reset()

            pbar = tqdm.tqdm(
                total=self.max_episode_steps,
                desc=f"Eval {policy_name} in RLBench::{self.env.env.env.task_name} {test_i+1}/{n_inits}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            while not done and pbar.n < pbar.total:
                # create obs dict
                obs_dict = dict_apply(
                    obs,
                    lambda x: torch.from_numpy(x).to(device=device, dtype=dtype).unsqueeze(0)
                )
                obs_dict = obs_dropout(obs_dict, self.obs_to_drop)

                # run policy
                with torch.inference_mode():
                    action = policy.predict_action({
                        port: obs_dict[port] 
                        for port in policy.get_observation_ports()
                    })['action'].detach().cpu().numpy().squeeze(0)

                # step env
                obs, reward, done, _, _ = self.env.step(action)

                all_success[test_i] = all_success[test_i] or (reward >= 1)

                # update pbar
                pbar.update(action.shape[0])
            pbar.close()

            # collect data for this round
            all_video_paths[test_i] = self.env.render()

        # clear out video buffer
        _ = self.env.reset()

        # log 
        log_data = dict()

        for task_name in set(self.env_task_names):
            task_success = [
                all_success[i]
                for i in range(n_inits)
                if self.env_task_names[i] == task_name
            ]
            log_data[f"{task_name}/mean_success_rate"] = np.mean(task_success)

        for i in range(n_inits):
            seed = self.env_seeds[i]
            task_name = self.env_task_names[i]

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                video = wandb.Video(video_path)
                log_data[f"{task_name}/video_{seed}"] = video
            
        # log aggregate metrics
        log_data['mean_success_rate'] = np.mean(all_success)
        
        return log_data


    def close(self):
        self.env.close()



_PARALLEL_RLBENCH = os.getenv("PARALLEL_RLBENCH", "1").lower() in ("1", "true", "yes")
if _PARALLEL_RLBENCH:
    RlbenchRunner = RlbenchParallelRunner
else:
    RlbenchRunner = RlbenchNonParallelRunner
    
    

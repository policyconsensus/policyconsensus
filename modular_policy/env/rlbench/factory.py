import numpy as np
from modular_policy.env.rlbench.env import RlbenchEnv
from typing import List, Optional

MT2_TASKS = [
    'close_box',
    'toilet_seat_down',
]

MT3_TASKS = [
    'close_box',
    'close_microwave',
    'toilet_seat_down',
]

MT4_TASKS = [
    "open_box",
    "toilet_seat_up",
    "open_drawer",
    "take_umbrella_out_of_umbrella_stand",
]

MT5_TASKS = [
    "open_box",
    "open_microwave",
    "toilet_seat_up",
    "open_drawer",
    "take_umbrella_out_of_umbrella_stand",
]

MT_TASKS = {
    'mt2': MT2_TASKS,
    'mt3': MT3_TASKS,
    'mt4': MT4_TASKS,
    'mt5': MT5_TASKS,
}


def is_multitask(task_name: str) -> bool:
    return task_name in MT_TASKS


def get_subtasks(task_name: str) -> List[str]:
    if is_multitask(task_name):
        return MT_TASKS[task_name]
    else:
        return [task_name]


def take_a_glance(task_names: List[str], 
    camera_name: Optional[str]) -> List[np.ndarray]:
    env = RlbenchEnv(image_size=512)
    obs = []
    for task_name in task_names:
        env.set_task(task_name)
        if camera_name is None:
            obs.append(env.render())
        else:
            obs.append(env.reset()[0][camera_name + '_rgb'].transpose(1, 2, 0))
    env.close()
    return obs

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os
import numpy as np
import tyro
import yaml
from gello.dynamixel.driver import DynamixelDriver

MENAGERIE_ROOT: Path = Path(__file__).parent / "third_party" / "mujoco_menagerie"


@dataclass
class Args:
    port: str = "/dev/ttyUSB0"
    """The port that GELLO is connected to."""

    start_joints: Tuple[float, ...] = (0, -1.57, 1.57, -1.57, -1.57, 1.57) #TODO
    
    # pillow and fruits
    # left arm (192.168.0.3): -3.1415926, -1.57, -1.57, -1.57, 1.57, 3.1415926
    # right arm (192.168.0.2): 3.1415926, -1.57, 1.57, -1.57, -1.57, -3.1415926
    
    
    # shelf picking
    # left arm (192.168.0.3): 0, -1.57, 1.57, -1.57, -1.57, 1.57
    # right arm (192.168.0.2): 0, -1.57, -1.57, -1.57, 1.57, 0

    """The joint angles that the GELLO is placed in at (in radians)."""

    joint_signs: Tuple[float, ...] = (1, 1, -1, 1, 1, 1)
    """The joint angles that the GELLO is placed in at (in radians)."""

    gripper: bool = True
    """Whether or not the gripper is attached."""
    
    bimanual: bool = False

    def __post_init__(self):
        assert len(self.joint_signs) == len(self.start_joints)
        for idx, j in enumerate(self.joint_signs):
            assert (
                j == -1 or j == 1
            ), f"Joint idx: {idx} should be -1 or 1, but got {j}."

    @property
    def num_robot_joints(self) -> int:
        return len(self.start_joints)

    @property
    def num_joints(self) -> int:
        extra_joints = 1 if self.gripper else 0
        return self.num_robot_joints + extra_joints

def get_gello_config(args, config, port_key, joints_key, side):
    args.port = config[port_key]
    args.start_joints = config[joints_key]
    args.joint_signs = config['joint_signs']
    print(f'offsets for {side} arm:')
    print(f'port: {args.port}')
    get_offset(args)  # Assuming get_config is defined elsewhere

def get_offset(args: Args) -> None:
    joint_ids = list(range(0, args.num_joints))
    driver = DynamixelDriver(joint_ids, port=args.port, baudrate=57600)

    # assume that the joint state shouold be args.start_joints
    # find the offset, which is a multiple of np.pi/2 that minimizes the error between the current joint state and args.start_joints
    # this is done by brute force, we seach in a range of +/- 8pi

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = args.joint_signs[index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = args.start_joints[index]
        return np.abs(joint_i - start_i)

    for _ in range(10):
        driver.get_joints()  # warmup

    for _ in range(1):
        best_offsets = []
        curr_joints = driver.get_joints()
        for i in range(args.num_robot_joints):
            best_offset = 0
            best_error = 1e6
            for offset in np.linspace(
                -8 * np.pi, 8 * np.pi, 8 * 4 + 1
            ):  # intervals of pi/2
                error = get_error(offset, i, curr_joints)
                if error < best_error:
                    best_error = error
                    best_offset = offset
            best_offsets.append(best_offset)
        # print()
        print("best offsets               : ", [f"{x:.3f}" for x in best_offsets])
        print("best offsets in degrees    : ["+ ", ".join([f"{int(np.rad2deg(x))}" for x in best_offsets]) + " ]")
        print(
            "best offsets function of pi: ["
            + ", ".join([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in best_offsets])
            + " ]",
        )
        if args.gripper:
            print(
                "gripper open (degrees)       ",
                np.rad2deg(driver.get_joints()[-1]) - 0.2,
            )
            print(
                "gripper close (degrees)      ",
                np.rad2deg(driver.get_joints()[-1]) - 42,
            )
        print()


def main(args: Args) -> None:
    if args.bimanual:
        gello_config_path = os.path.join(Path(__file__).parent.parent.parent.parent, "config/teleop/gello.yaml")
        
        with open(gello_config_path) as stream:
            try:
                gello_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                
        get_gello_config(args, gello_config, 'left_gello_port', 'left_start_joints', 'left')
        get_gello_config(args, gello_config, 'right_gello_port', 'right_start_joints', 'right')


    else:
        get_offset(args)


if __name__ == "__main__":
    main(tyro.cli(Args))

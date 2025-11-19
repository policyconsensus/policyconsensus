import numpy as np
from rlbench.backend.scene import Scene
from rlbench.action_modes.action_mode import ActionMode
from rlbench.action_modes.arm_action_modes import JointPosition
from rlbench.action_modes.gripper_action_modes import (
    GripperActionMode, assert_action_shape)



class GripperJointPosition(GripperActionMode):

    def __init__(self, attach_grasped_objects: bool = True,
                 detach_before_open: bool = True,
                 absolute_mode: bool = True):
        self._attach_grasped_objects = attach_grasped_objects
        self._detach_before_open = detach_before_open
        self._absolute_mode = absolute_mode
        self._control_mode_set = False

    def action(self, scene: Scene, action: np.ndarray):
        self.action_pre_step(scene, action)
        self.action_step(scene, action)
        self.action_post_step(scene, action)

    def action_pre_step(self, scene: Scene, action: np.ndarray):
        if not self._control_mode_set:
            scene.robot.gripper.set_control_loop_enabled(True)
            self._control_mode_set = True
        assert_action_shape(action, self.action_shape(scene.robot))
        a = action if self._absolute_mode else (
            action + scene.robot.gripper.get_joint_positions())
        scene.robot.gripper.set_joint_target_positions(a)

    def action_step(self, scene: Scene, action: np.ndarray):
        scene.step()

    def action_post_step(self, scene: Scene, action: np.ndarray):
        scene.robot.gripper.set_joint_target_positions(
            scene.robot.gripper.get_joint_positions())

    def action_shape(self, scene: Scene) -> tuple:
        return 2,



class JointPositionActionMode(ActionMode):
    """A pre-set, delta joint position action mode or arm and abs for gripper.

    Both the arm and gripper action are applied at the same time.
    """

    def __init__(self):
        super(JointPositionActionMode, self).__init__(
            arm_action_mode=JointPosition(absolute_mode=True),
            gripper_action_mode=GripperJointPosition(absolute_mode=True)
        )

    def action(self, scene: Scene, action: np.ndarray):
        arm_act_size = np.prod(self.arm_action_mode.action_shape(scene))
        arm_action = np.array(action[:arm_act_size])
        ee_action = np.array(action[arm_act_size:])
        self.arm_action_mode.action_pre_step(scene, arm_action)
        self.gripper_action_mode.action_pre_step(scene, ee_action)
        scene.step()
        self.arm_action_mode.action_post_step(scene, arm_action)
        self.gripper_action_mode.action_post_step(scene, ee_action)

    def action_shape(self, scene: Scene):
        return np.prod(self.arm_action_mode.action_shape(scene)) + np.prod(
            self.gripper_action_mode.action_shape(scene))

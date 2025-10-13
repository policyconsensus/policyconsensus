from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si
import scipy.spatial.transform as st

def rotation_distance(a: st.Rotation, b: st.Rotation) -> float:
    """
    Calculate the rotational distance between two rotations.

    Args:
        a (st.Rotation): The first rotation.
        b (st.Rotation): The second rotation.

    Returns:
        float: The magnitude of the rotation from 'a' to 'b'.
    """
    return (b * a.inv()).magnitude()

def pose_distance(start_pose, end_pose):
    """
    Calculate the positional and rotational distance between two poses.

    Args:
        start_pose: The starting pose (position + rotation vector).
        end_pose: The ending pose (position + rotation vector).

    Returns:
        A tuple containing the positional and rotational distances.
    """
    start_pose = np.array(start_pose)
    end_pose = np.array(end_pose)
    start_pos = start_pose[:3]
    end_pos = end_pose[:3]
    # Handle both 6D and 7D poses (7D includes gripper as last dimension)
    start_rot = st.Rotation.from_rotvec(start_pose[3:6])
    end_rot = st.Rotation.from_rotvec(end_pose[3:6])
    pos_dist = np.linalg.norm(end_pos - start_pos)
    rot_dist = rotation_distance(start_rot, end_rot)
    return pos_dist, rot_dist

class PoseTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, poses: np.ndarray):
        assert len(times) >= 1
        assert len(poses) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(poses, np.ndarray):
            poses = np.array(poses)

        # Determine if poses are 6D or 7D
        self.pose_dim = poses.shape[1] if poses.ndim > 1 else len(poses[0])
        self.has_gripper = (self.pose_dim == 7)

        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._poses = poses
            # For single step, store gripper values if present
            if self.has_gripper:
                self.gripper_values = poses[:,6] if poses.ndim > 1 else [poses[0][6]]
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            pos = poses[:,:3]
            # Handle both 6D and 7D poses (7D includes gripper as last dimension)
            rot = st.Rotation.from_rotvec(poses[:,3:6])

            self.pos_interp = si.interp1d(times, pos, 
                axis=0, assume_sorted=True)
            self.rot_interp = st.Slerp(times, rot)
            
            # Store gripper values if present (no interpolation needed, use target value)
            if self.has_gripper:
                self.gripper_values = poses[:,6]
    
    @property
    def times(self) -> np.ndarray:
        """
        Get the array of times associated with the poses.

        Returns:
            np.ndarray: An array of times.
        """
        if self.single_step:
            return self._times
        else:
            return self.pos_interp.x
    
    @property
    def poses(self) -> np.ndarray:
        """
        Get the array of poses associated with the times.

        Returns:
            np.ndarray: An array of poses (positions and rotations).
        """
        if self.single_step:
            return self._poses
        else:
            n = len(self.times)
            poses = np.zeros((n, self.pose_dim))
            poses[:,:3] = self.pos_interp.y
            poses[:,3:6] = self.rot_interp(self.times).as_rotvec()
            
            # Add gripper values if present
            if self.has_gripper:
                # Use the last gripper value for all poses (no interpolation)
                poses[:,6] = self.gripper_values[-1]
            
            return poses

    def trim(self, 
            start_t: float, end_t: float
            ) -> "PoseTrajectoryInterpolator":
        """
        Trim the interpolator to a specified time range.

        Args:
            start_t (float): The start time of the new range.
            end_t (float): The end time of the new range.

        Returns:
            PoseTrajectoryInterpolator: A new interpolator within the specified time range.
        """
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_poses = self(all_times)
        return PoseTrajectoryInterpolator(times=all_times, poses=all_poses)
    
    def drive_to_waypoint(self, 
            pose, time, curr_time,
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf
        ) -> "PoseTrajectoryInterpolator":
        """
        Drive the trajectory to a new waypoint.

        Args:
            pose (np.ndarray): The target pose.
            time (float): The time at which the target pose should be reached.
            curr_time (float): The current time.
            max_pos_speed (float, optional): Maximum positional speed. Defaults to np.inf.
            max_rot_speed (float, optional): Maximum rotational speed. Defaults to np.inf.

        Returns:
            PoseTrajectoryInterpolator: A new PoseTrajectoryInterpolator with the added waypoint.
        """
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        time = max(time, curr_time)
        
        curr_pose = self(curr_time)
        pos_dist, rot_dist = pose_distance(curr_pose, pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = time - curr_time
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new pose
        trimmed_interp = self.trim(curr_time, curr_time)
        
        # Debug: Print dimensions
        print(f"DEBUG drive_to_waypoint:")
        print(f"  self.pose_dim: {self.pose_dim}, self.has_gripper: {self.has_gripper}")
        print(f"  trimmed_interp.pose_dim: {trimmed_interp.pose_dim}, trimmed_interp.has_gripper: {trimmed_interp.has_gripper}")
        print(f"  trimmed_interp.poses.shape: {trimmed_interp.poses.shape}")
        print(f"  pose.shape: {np.array(pose).shape}")
        
        # Handle dimension mismatch between trimmed interpolator and new pose
        pose_array = np.array(pose)
        if trimmed_interp.pose_dim != len(pose_array):
            if trimmed_interp.pose_dim == 6 and len(pose_array) == 7:
                # Truncate gripper dimension from incoming pose
                pose_array = pose_array[:6]
            elif trimmed_interp.pose_dim == 7 and len(pose_array) == 6:
                # Add gripper dimension (default to 0)
                pose_array = np.append(pose_array, [0])
        
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose_array], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp

    def schedule_waypoint(self,
            pose, time, 
            max_pos_speed=np.inf, 
            max_rot_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "PoseTrajectoryInterpolator":
        """
        Schedule a new waypoint into the trajectory.

        Args:
            pose (np.ndarray): The target pose.
            time (float): The time at which the target pose should be reached.
            max_pos_speed (float, optional): Maximum positional speed. Defaults to np.inf.
            max_rot_speed (float, optional): Maximum rotational speed. Defaults to np.inf.
            curr_time (float, optional): The current time. Defaults to None.
            last_waypoint_time (float, optional): The last waypoint time. Defaults to None.

        Returns:
            PoseTrajectoryInterpolator: A new PoseTrajectoryInterpolator with the scheduled waypoint.
        """
        assert(max_pos_speed > 0)
        assert(max_rot_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None

        # trim current interpolator to between curr_time and last_waypoint_time
        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time

        if curr_time is not None:
            if time <= curr_time:
                # if insert time is earlier than current time
                # no effect should be done to the interpolator
                return self
            # now, curr_time < time
            start_time = max(curr_time, start_time)

            if last_waypoint_time is not None:
                # if last_waypoint_time is earlier than start_time
                # use start_time
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        # end time should be the latest of all times except time
        # after this we can assume order (proven by zhenjia, due to the 2 min operations)

        # Constraints:
        # start_time <= end_time <= time (proven by zhenjia)
        # curr_time <= start_time (proven by zhenjia)
        # curr_time <= time (proven by zhenjia)
        
        # time can't change
        # last_waypoint_time can't change
        # curr_time can't change
        assert start_time <= end_time
        assert end_time <= time
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)

        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        trimmed_interp = self.trim(start_time, end_time)
        # after this, all waypoints in trimmed_interp is within start_time and end_time
        # and is earlier than time

        # determine speed
        duration = time - end_time
        end_pose = trimmed_interp(end_time)
        pos_dist, rot_dist = pose_distance(pose, end_pose)
        pos_min_duration = pos_dist / max_pos_speed
        rot_min_duration = rot_dist / max_rot_speed
        duration = max(duration, max(pos_min_duration, rot_min_duration))
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new pose
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        poses = np.append(trimmed_interp.poses, [pose], axis=0)

        # create new interpolator
        final_interp = PoseTrajectoryInterpolator(times, poses)
        return final_interp


    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        """
        Evaluate the pose at a given time or times.

        Args:
            t (Union[numbers.Number, np.ndarray]): The time(s) at which to evaluate the pose.

        Returns:
            np.ndarray: The pose at the specified time(s).
        """
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        pose = np.zeros((len(t), self.pose_dim))
        if self.single_step:
            pose[:] = self._poses[0]
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            pose[:,:3] = self.pos_interp(t)
            pose[:,3:6] = self.rot_interp(t).as_rotvec()
            
            # Add gripper values if present
            if self.has_gripper:
                # Use the last gripper value for all poses (no interpolation)
                pose[:,6] = self.gripper_values[-1]

        if is_single:
            pose = pose[0]
        return pose

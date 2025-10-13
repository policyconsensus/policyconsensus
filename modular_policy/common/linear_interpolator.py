from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si

class LinearInterpolator:
    def __init__(self, times: np.ndarray, cmds: np.ndarray):
        """Initialize a linear interpolator

        Args:
            times (np.ndarray): (t,) array of times
            cmds (np.ndarray): (t, d) array of commands
        """
        assert len(times) >= 1
        assert len(cmds) == len(times)
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(cmds, np.ndarray):
            cmds = np.array(cmds)

        self.cmd_dim = cmds[0].shape[0]
        if len(times) == 1:
            # special treatment for single step interpolation
            self.single_step = True
            self._times = times
            self._cmds = cmds
        else:
            self.single_step = False
            assert np.all(times[1:] >= times[:-1])

            self.cmd_interp = si.interp1d(times, cmds, 
                axis=0, assume_sorted=True)
    
    @property
    def times(self) -> np.ndarray:
        if self.single_step:
            return self._times
        else:
            return self.cmd_interp.x
    
    @property
    def cmds(self) -> np.ndarray:
        if self.single_step:
            return self._cmds
        else:
            return self.cmd_interp.y

    def trim(self, 
            start_t: float, end_t: float
            ) -> "LinearInterpolator":
        assert start_t <= end_t
        times = self.times
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # remove duplicates, Slerp requires strictly increasing x
        all_times = np.unique(all_times)
        # interpolate
        all_cmds = self(all_times)
        return LinearInterpolator(times=all_times, cmds=all_cmds)
    
    def drive_to_waypoint(self, 
            cmd, time, curr_time,
            max_delta_cmd=np.inf,
        ) -> "LinearInterpolator":
        assert(max_delta_cmd > 0)
        time = max(time, curr_time)
        
        curr_cmd = self(curr_time)
        cmd_dist = np.linalg.norm(cmd - curr_cmd)
        cmd_min_duration = cmd_dist / max_delta_cmd
        duration = time - curr_time
        duration = max(duration, cmd_min_duration)
        assert duration >= 0
        last_waypoint_time = curr_time + duration

        # insert new cmd
        trimmed_interp = self.trim(curr_time, curr_time)
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        cmds = np.append(trimmed_interp.cmds, [cmd], axis=0)

        # create new interpolator
        final_interp = LinearInterpolator(times=times, cmds=cmds)
        return final_interp

    def schedule_waypoint(self,
            cmd, time, 
            max_cmd_speed=np.inf,
            curr_time=None,
            last_waypoint_time=None
        ) -> "LinearInterpolator":
        """Schedule a waypoint at time with cmd

        Args:
            cmd (np.ndarray): (d,) array of command
            time (float): time to schedule waypoint
            max_cmd_speed (float, optional): maximum speed of the command. Defaults to np.inf.
            curr_time (float, optional): current time. Defaults to None.
            last_waypoint_time (float, optional): last waypoint time. Defaults to None.

        Returns:
            LinearInterpolator: new interpolator with scheduled waypoint
        """
        assert(max_cmd_speed > 0)
        if last_waypoint_time is not None:
            assert curr_time is not None
        assert cmd.shape[0] == self.cmd_dim

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
        end_cmd = trimmed_interp(end_time)
        cmd_dist = np.linalg.norm(cmd - end_cmd)
        cmd_min_duration = cmd_dist / max_cmd_speed
        duration = max(duration, cmd_min_duration)
        assert duration >= 0
        last_waypoint_time = end_time + duration

        # insert new cmd
        times = np.append(trimmed_interp.times, [last_waypoint_time], axis=0)
        cmds = np.append(trimmed_interp.cmds, [cmd], axis=0)

        # create new interpolator
        final_interp = LinearInterpolator(times=times, cmds=cmds)
        return final_interp


    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:
        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        cmd = None
        if self.single_step:
            cmd = self._cmds
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)

            cmd = self.cmd_interp(t)

        if is_single:
            return cmd[0]

        return cmd
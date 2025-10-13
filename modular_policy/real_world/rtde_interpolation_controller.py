import os
import sys
import time
import enum
import warnings
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
from modular_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from modular_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from modular_policy.shared_memory.shared_ndarray import SharedNDArray
from modular_policy.common.trajectory_interpolator import (
    PoseTrajectoryInterpolator, JointTrajectoryInterpolator)

class Command(enum.Enum):
    STOP = 0
    SERVO_EE_POSE = 1
    SCHEDULE_WAYPOINT = 2
    SERVOJ = 3
    SCHEDULE_JOINTS = 4
    GET_FK = 5
    GET_IK = 6
    SET_FREEDRIVE = 7
    EEF_MODE = 8
    JOINT_MODE = 9


class RTDEInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip, 
        frequency=125, 
        lookahead_time=0.1, 
        gain=100,
        max_pos_speed=0.25, # 5% of max speed
        max_rot_speed=0.16, # 5% of max speed
        speed_slider_value=0.1,
        launch_timeout=3,
        tcp_offset_pose=None,
        payload_mass=None,
        payload_cog=None,
        joints_init=None,
        joints_init_speed=0.5,
        soft_real_time=False,
        verbose=False,
        receive_keys=None,
        get_max_k=128,
        dummy_robot=False,
        use_gripper=False,
        extrinsics_dir=None,
        extrinsics_name=None,
        ctrl_mode='joint'
    ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        assert 100 <= gain <= 2000
        assert 0 < max_pos_speed
        assert 0 < max_rot_speed
        if tcp_offset_pose is not None:
            tcp_offset_pose = np.array(tcp_offset_pose)
            assert tcp_offset_pose.shape == (6,)
        if payload_mass is not None:
            assert 0 <= payload_mass <= 5
        if payload_cog is not None:
            payload_cog = np.array(payload_cog)
            assert payload_cog.shape == (3,)
            assert payload_mass is not None
        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,)

        super().__init__(name="RTDEPositionalController")
        self.dummy_robot = dummy_robot
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.speed_slider_value = speed_slider_value
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.ctrl_mode = ctrl_mode
        
        self.num_joints = 6 + int(use_gripper)
        
        self.use_gripper = use_gripper
        if self.use_gripper:
            # install by going to `/home/haonan/Projects/direct_human_demo/ti/devices/gello_software`
            # and run `pip install -e .`
            from gello.robots.robotiq_gripper import RobotiqGripper

            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=robot_ip, port=63352)
            print("gripper connected")

        if extrinsics_dir is None or extrinsics_name is None:
            self.base_pose_in_world = np.eye(4)
            if extrinsics_dir is None and extrinsics_name is None:
                pass  # Expected default behavior
            else:
                warnings.warn("Both extrinsics_dir and extrinsics_name must be provided, using identity matrix as base pose in world")
        else:
            extrinsics_path = os.path.join(extrinsics_dir, extrinsics_name)
            if not os.path.exists(extrinsics_path):
                self.base_pose_in_world = np.eye(4)
                warnings.warn(f"extrinsics_path {extrinsics_path} does not exist, using identity matrix as base pose in world")
            else:
                self.base_pose_in_world = np.load(extrinsics_path)

        # build input queue
        example = {
            'cmd': Command.SERVOJ.value,
            'target_ee_pose': np.zeros((self.num_joints,), dtype=np.float64),
            'target_joints': np.zeros((self.num_joints,), dtype=np.float64), 
            'joints': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0,
            'speed': 0.0,
            'acceleration': 0.0,
            'asynchronous': False,
            'free_drive': 0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        if receive_keys is None:
            receive_keys = [
                'ActualTCPPose',
                'ActualTCPSpeed',
                'ActualQ',
                'ActualQd',

                'TargetTCPPose',
                'TargetTCPSpeed',
                'TargetQ',
                'TargetQd',
                
                'FtRawWrench'
            ]
        example = dict()
        if not self.dummy_robot:
            rtde_r = RTDEReceiveInterface(hostname=robot_ip)
            for key in receive_keys:
                example[key] = np.array(getattr(rtde_r, 'get'+key)())
                if  self.use_gripper:
                    gripper_pos = self._get_gripper_pos()
                    example['gripper_pos'] = gripper_pos
                    if key == 'ActualQ' or key == 'ActualTCPPose' or key == 'TargetQ' or key == 'TargetTCPPose':
                        example[key] = np.append(example[key], gripper_pos)

        example['robot_receive_timestamp'] = time.time()
        example['robot_base_pose_in_world'] = self.base_pose_in_world
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        
        self.rtde_c = None  # Initialize rtde_c as None

        # self.response_queue = mp.Queue()
        self.parent_conn, self.child_conn = mp.Pipe()
    
        # create shared array for intrinsics
        fk_result = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(6+int(self.use_gripper),),
                dtype=np.float64)
        self.fk_result = fk_result
        
        ik_result = SharedNDArray.create_from_shape(   
                mem_mgr=shm_manager,
                shape=(6+int(self.use_gripper),),
                dtype=np.float64)
        self.ik_result = ik_result
        

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if self.dummy_robot:
            return True
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        if self.dummy_robot:
            return True
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        if self.dummy_robot:
            return True

        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
        if self.dummy_robot:
            return True

    @property
    def is_ready(self):
        if self.dummy_robot:
            return True

        return self.ready_event.is_set()
    
    # ========= mode switching =========
    def switch_mode(self, mode: str):
        if self.dummy_robot:
            return True
        assert self.is_alive()
        if self.ctrl_mode is None or mode != self.ctrl_mode:
            if mode == 'eef':
                message = {
                    'cmd': Command.EEF_MODE.value
                }
                self.ctrl_mode = 'eef'
            elif mode == 'joint':
                message = {
                    'cmd': Command.JOINT_MODE.value
                }
                self.ctrl_mode = 'joint'
            else:
                raise ValueError(f"mode {mode} not recognized")
            self.input_queue.put(message)

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def servo_ee_pose(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.ctrl_mode == 'eef'
        if self.dummy_robot:
            return True
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6 + int(self.use_gripper),)

        target_joint_pos = np.zeros((self.num_joints,), dtype=np.float64)
            
        message = {
            'cmd': Command.SERVO_EE_POSE.value,
            'target_joints': target_joint_pos, # dummy value
            'target_ee_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
        
    def servoJ(self, qpos, duration=0.1):
        """
        duration: desired time to reach qpos
        """
        assert self.ctrl_mode == 'joint'
        if self.dummy_robot:
            return True
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        qpos = np.array(qpos)
        if qpos.shape == (6,) and self.use_gripper:
            qpos = np.append(qpos, 0)
        assert qpos.shape == (self.num_joints,) 

        message = {
            'cmd': Command.SERVOJ.value,
            'target_joints': qpos,
            'target_ee_pose': np.zeros((6 + int(self.use_gripper)), dtype=np.float64), # 'target_ee_pose' is not used in this case, so we just put a dummy value here,
            'duration': duration
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time):
        assert target_time > time.time()
        pose = np.array(pose)
        assert pose.shape == (6 + int(self.use_gripper),)

        if self.ctrl_mode == 'joint':
            joint = self.get_inverse_kinematics(pose)
            message = {
                'cmd': Command.SCHEDULE_JOINTS.value,
                'target_joints': joint,
                'target_time': target_time
            }
        else:
            message = {
                'cmd': Command.SCHEDULE_WAYPOINT.value,
                'target_ee_pose': pose,
                'target_time': target_time
            }
        self.input_queue.put(message)

    def schedule_joints(self, qpos, target_time):
        assert target_time > time.time()
        qpos = np.array(qpos)
        assert qpos.shape == (self.num_joints,)
        
        if self.ctrl_mode == 'eef':
            pose = self.get_forward_kinematics(qpos)
            message = {
                'cmd': Command.SCHEDULE_WAYPOINT.value,
                'target_ee_pose': pose,
                'target_time': target_time
            }
        else:
            message = {
            'cmd': Command.SCHEDULE_JOINTS.value,
            'target_joints': qpos,
            'target_time': target_time
            }
        self.input_queue.put(message)
        

    def get_forward_kinematics(self, joints):    
        robot_joints = joints[:6]
        if self.use_gripper:
            gripper_joints = joints[6]        
        # Send the joints to the child process for FK calculation
        self.parent_conn.send(('GET_FK', robot_joints))
        # Wait for the result
        while True:
            if self.parent_conn.poll():
                msg = self.parent_conn.recv()
                if msg == "FK_RESULT_READY":
                    break
        if self.use_gripper:
            self.fk_result.get()[-1] = gripper_joints 
        return self.fk_result.get()[:]
    
    def get_inverse_kinematics(self, eef_pose):    
        pose = eef_pose[:6]
        if self.use_gripper:
            gripper_joints = eef_pose[6]        
        # Send the joints to the child process for FK calculation
        self.parent_conn.send(('GET_IK', pose))
        # Wait for the result
        while True:
            if self.parent_conn.poll():
                msg = self.parent_conn.recv()
                if msg == "FK_RESULT_READY":
                    break
        if self.use_gripper:
            self.ik_result.get()[-1] = gripper_joints 
        return self.ik_result.get()[:]

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= helper function ===========
    def _get_gripper_pos(self) -> float:
        # import time

        # time.sleep(0.01)
        gripper_pos = self.gripper.get_current_position()
        assert 0 <= gripper_pos <= 255, "Gripper position must be between 0 and 255"
        return gripper_pos / 255

    def set_robot_joints(self, set_joints, speed=0.004):
        """
        Resets the robot joints to a specified configuration.

        Parameters:
        - env: The environment object containing the robot.
        - set_joints: The target joint positions in radians.
        """
        reset_dt = 0.1
        curr_joints = self.get_state()['ActualQ']
        dof = set_joints.shape[0]
        max_delta = np.abs(curr_joints[:dof] - set_joints).max()
        
        # steps = min(int(max_delta / 0.0005), 1000)
        
        # Estimate the total time needed to cover the maximum distance at the given speed
        total_time = max_delta / speed
        
        # Define a minimum duration for each command to ensure smooth movement
        min_duration = 0.1
        
        # Calculate the number of steps needed based on the minimum duration
        steps = max(int(total_time / min_duration), 1)

        for jnt in np.linspace(curr_joints[:dof], set_joints, steps):
            self.servoJ(jnt, duration=reset_dt)
            time.sleep(0.01) # used to give the rtde process enough time to execute the command

    # ========= main loop in process ============
    def run(self):
        if self.dummy_robot:
            return
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start rtde
        robot_ip = self.robot_ip
        self.rtde_r = RTDEReceiveInterface(hostname=robot_ip)
        self.rtde_c = RTDEControlInterface(hostname=robot_ip)
        try:
            self.rtde_io_interface = RTDEIOInterface(robot_ip)
            self.rtde_io_interface.setSpeedSlider(self.speed_slider_value)
        except:
            print("Failed to connect to RTDE IO interface")
            print("Speed slider value will not be set")
        
        self.rtde_c.endFreedriveMode()

        try:
            # set parameters
            if self.tcp_offset_pose is not None:
                self.rtde_c.setTcp(self.tcp_offset_pose)
            else:
                self.tcp_offset_pose = self.rtde_c.getTCPOffset()

            if self.payload_mass is not None:
                if self.payload_cog is not None:
                    assert self.rtde_c.setPayload(self.payload_mass, self.payload_cog)
                else:
                    assert self.rtde_c.setPayload(self.payload_mass)
            
            assert self.rtde_c.zeroFtSensor()

            # main loop
            dt = 1. / self.frequency
            curr_ee_pose = self.rtde_r.getActualTCPPose()
            curr_joint_pos = self.rtde_r.getActualQ()
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t

            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_ee_pose]
            )
            joint_interp = JointTrajectoryInterpolator(
                times=[curr_t],
                qpos=[curr_joint_pos]
            )
            if self.use_gripper:
                curr_gripper_pos = self._get_gripper_pos()
                gripper_interp = JointTrajectoryInterpolator(
                    times=[curr_t],
                    qpos=[np.array([curr_gripper_pos])]
                )

            iter_idx = 0
            keep_running = True
            while keep_running:
                if self.child_conn.poll():  # Check if there's data
                    cmd, data = self.child_conn.recv()  # Receive the command and data
                    if cmd == 'GET_FK':
                        joints = data
                        # Write the result to shared memory
                        self.fk_result.get()[:6] = self.rtde_c.getForwardKinematics(joints, self.tcp_offset_pose)
                        # Notify the parent process that new data is available
                        self.child_conn.send("FK_RESULT_READY")
                    elif cmd == 'GET_IK':
                        pose = data
                        # Write the result to shared memory
                        self.ik_result.get()[:6] = self.rtde_c.getInverseKinematics(pose)
                        # Notify the parent process that new data is available
                        self.child_conn.send("FK_RESULT_READY")

                # start control iteration
                t_start = self.rtde_c.initPeriod()

                # send command to robot
                t_now = time.monotonic()
                
                vel = 0.5 # dummy vel and acc, not used by ur5 in current version
                acc = 0.5 # dummy vel and acc, not used by ur5 in current version

                if self.ctrl_mode == 'eef':
                    # read from interpolation
                    eef_pose_command = pose_interp(t_now)
                    # inverse kinematics
                    pose_command = self.rtde_c.getInverseKinematics(eef_pose_command)

                else:
                    pose_command = joint_interp(t_now)

                # https://sdurobotics.gitlab.io/ur_rtde/api/api.html#_CPPv4N7ur_rtde20RTDEControlInterface6servoLERKNSt6vectorIdEEddddd
                assert self.rtde_c.servoJ(pose_command,  
                    vel, acc,  # dummy vel and acc, not used by ur5 in current version
                    dt, # time where the command is controlling the robot. The function is blocking for time t [S]
                    self.lookahead_time, 
                    self.gain)

                if self.use_gripper:
                    gripper_pos = gripper_interp(t_now)
                    gripper_pos = int(gripper_pos * 255)
                    gripper_pos = np.clip(gripper_pos, 0, 255)
                    self.gripper.move(gripper_pos, 255, 10)

                # update robot state
                state = dict()
                for key in self.receive_keys:
                    state[key] = np.array(getattr(self.rtde_r, 'get'+key)())
                    if  self.use_gripper:
                        gripper_pos = self._get_gripper_pos()
                        state['gripper_pos'] = gripper_pos
                        if key == 'ActualQ' or key == 'ActualTCPPose' or key == 'TargetQ' or key == 'TargetTCPPose':
                            state[key] = np.append(state[key], gripper_pos)

                state['robot_receive_timestamp'] = time.time()
                state['robot_base_pose_in_world'] = self.base_pose_in_world

                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break

                    elif cmd == Command.EEF_MODE.value:
                        self.ctrl_mode = 'eef'
                        curr_ee_pose = self.rtde_r.getActualTCPPose()
                        curr_t = time.monotonic()
                        pose_interp = PoseTrajectoryInterpolator(
                            times=[curr_t],
                            poses=[curr_ee_pose]
                        )

                    elif cmd == Command.JOINT_MODE.value:
                        self.ctrl_mode = 'joint'
                        curr_joint_pos = self.rtde_r.getActualQ()
                        curr_t = time.monotonic()
                        joint_interp = JointTrajectoryInterpolator(
                            times=[curr_t],
                            qpos=[curr_joint_pos]
                        )

                    elif cmd == Command.SERVO_EE_POSE.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_ee_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                            
                    elif cmd == Command.SERVOJ.value:
                        target_joints = command['target_joints']
                        duration = float(command['duration'])

                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        joint_interp = joint_interp.drive_to_waypoint(
                            qpos=target_joints[:6],
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        
                        if self.use_gripper:
                            gripper_interp = gripper_interp.schedule_waypoint(
                                qpos=np.array([target_joints[-1]]),
                                time=t_insert,
                                max_qpos_speed=5,
                                curr_time=curr_time
                            )

                        last_waypoint_time = curr_time + duration

                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))

                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_ee_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        assert isinstance(pose_interp, PoseTrajectoryInterpolator), type(pose_interp)
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose[:6],
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        if self.use_gripper:
                            gripper_interp = gripper_interp.schedule_waypoint(
                                qpos=np.array([target_pose[-1]]),
                                time=target_time,
                                max_qpos_speed=5,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                        last_waypoint_time = target_time

                    elif cmd == Command.SCHEDULE_JOINTS.value:
                        target_joints = command['target_joints']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        assert isinstance(joint_interp, JointTrajectoryInterpolator), type(joint_interp)
                        joint_interp = joint_interp.schedule_waypoint(
                            qpos=target_joints[:6],
                            time=target_time,
                            max_qpos_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        if self.use_gripper:
                            gripper_interp = gripper_interp.schedule_waypoint(
                                qpos=np.array([target_joints[-1]]),
                                time=target_time,
                                max_qpos_speed=5,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                        last_waypoint_time = target_time

                    else:
                        keep_running = False
                        break
                

                # regulate frequency
                self.rtde_c.waitPeriod(t_start)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[RTDEPositionalController] Actual frequency {1/(time.perf_counter() - t_start)}")

        finally:
            # manditory cleanup
            # decelerate
            self.rtde_c.servoStop()

            # terminate
            self.rtde_c.stopScript()
            self.rtde_c.disconnect()
            self.rtde_r.disconnect()
            self.ready_event.set()

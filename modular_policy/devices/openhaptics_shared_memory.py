import sys
sys.path.insert(1, '.')


import multiprocessing as mp
import numpy as np
import time
from atm.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from atm.common.trans_utils import matrix_to_euler
from pyOpenHaptics.hd_device import HapticDevice
import pyOpenHaptics.hd as hd
import time, os
from dataclasses import dataclass, field
from pyOpenHaptics.hd_callback import hd_callback
from multiprocessing.managers import SharedMemoryManager
import functools
from atm.shared_memory.shared_ndarray import SharedNDArray

@dataclass
class DeviceState:
    button: int = 0
    position: list = field(default_factory=list)
    rotation: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    gimbals: list = field(default_factory=list)
    force: list = field(default_factory=list)


class Openhaptics(mp.Process):

    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=200,
            max_value=500, 
            deadzone=(0,0,0,0,0,0), 
            dtype=np.float32,
            n_buttons=2,
            verbose=False,
            ):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        super().__init__()
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # copied variables
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        # self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
        # self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

        example = {
            # 3 translation, 3 rotation, 1 period
            'motion_event': np.zeros((7,), dtype=np.int64),
            # left and right button
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'joint_state': np.zeros((6,), dtype=np.float32),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer
        
        self.verbose = verbose

        button_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(1,),
                dtype=np.float64)
        button_array.get()[:] = 0

        position_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(3,),
                dtype=np.float64)
        position_array.get()[:] = 0

        rotation_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(3,),
                dtype=np.float64)
        rotation_array.get()[:] = 0

        joints_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(6,),
                dtype=np.float64)
        joints_array.get()[:] = 0

        force_array = SharedNDArray.create_from_shape(
                mem_mgr=shm_manager,
                shape=(3,),
                dtype=np.float64)
        force_array.get()[:] = 0
        
        self.button_array = button_array
        self.position_array = position_array
        self.rotation_array = rotation_array
        self.joints_array = joints_array
        self.force_array = force_array


    # ======= get state APIs ==========

    def get_motion_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['motion_event'][:6], 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state

    def get_joint_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['joint_state'][:6], 
            dtype=np.float32)

        return state
    
    def get_button_state(self):
        state = self.ring_buffer.get()
        return state['button_state']
    
    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]
    
    #========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    
    def run(self):
        if self.verbose:
            print("Haptic device process started.")
            
        device_state = DeviceState()      

        @hd_callback
        def state_callback():
            transform = np.asarray(hd.get_transform())
            joints = np.asarray(hd.get_joints())
            gimbals = np.asarray(hd.get_gimbals())

            # Extract rotation matrix (upper 3x3 of transform)
            rotation_matrix = [row[:3] for row in transform[:3]]
            # Convert rotation matrix to Euler angles (or use quaternions)
            euler_angles = matrix_to_euler(rotation_matrix)

            self.position_array.get()[:] = [transform[3][0], -transform[3][1], transform[3][2]]
            self.rotation_array.get()[:] = euler_angles  # Update this line to set the rotation
         
            self.joints_array.get()[:3] = joints

            self.joints_array.get()[3:] = gimbals
            
            hd.set_force(self.force_array.get()[:])
            
            self.button_array.get()[:] = hd.get_buttons() # can be 0, 1, 2

        
        device = HapticDevice(device_name="left", callback=state_callback)

        # device = HapticDevice(device_name="left", callback=state_callback)
        try:
            motion_event = np.zeros((7,), dtype=np.int64)
            joint_state = np.zeros((6,), dtype=np.float32)
            button_state = np.zeros((self.n_buttons,), dtype=bool)
            # send one message immediately so client can start reading
            self.ring_buffer.put({
                'motion_event': motion_event,
                'joint_state': joint_state,
                'button_state': button_state,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():

                if self.verbose:
                    print("Polling Haptic device event...")

                receive_timestamp = time.time()
                motion_event[:3] = self.position_array.get()[:]
                motion_event[3:6] = self.rotation_array.get()[:]
                motion_event[6] = 0
                
                joint_state[:] = self.joints_array.get()[:]
                
                button_array = self.button_array.get()[:]
                if button_array == 1:
                    button_state[:] = np.array([1, 0])
                elif button_array == 2:
                    button_state[:] = np.array([0, 1])

                    
                # finish integrating this round of events
                # before sending over
                self.ring_buffer.put({
                    'motion_event': motion_event,
                    'joint_state': joint_state,
                    'button_state': button_state,
                    'receive_timestamp': receive_timestamp
                })
                time.sleep(1/self.frequency)
        finally:
            device.close()
            if self.verbose:
                print("Space Mouse process stopped.")


def get_haptics_offset(joint_sign):
    def get_error(offset: float, index: int, joint_state: np.ndarray, joint_sign) -> float:
        joint_i = joint_sign[i]*(joint_state[index] - offset)
        start_i = robot_joints[index]
        return np.abs(joint_i - start_i)

    with SharedMemoryManager() as shm_manager:
        with Openhaptics(shm_manager, verbose=False) as haptic_device:
            joint_pos = haptic_device.get_joint_state()
            
            robot_joints = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])
            
            num_joint = 6
            
            best_offsets = []

            for i in range(num_joint):
                best_offset = 0
                best_error = 1e6
                for offset in np.linspace(
                    -8 * np.pi, 8 * np.pi, 8 * 4 + 1
                ):  # intervals of pi/2
                    error = get_error(offset, i, joint_pos, joint_sign)
                    if error < best_error:
                        best_error = error
                        best_offset = offset
                best_offsets.append(best_offset)
            print()
            print("best offsets               : ", [f"{x:.3f}" for x in best_offsets])
            print(
                "best offsets function of pi: ["
                + ", ".join([f"{int(np.round(x/(np.pi/2)))}*np.pi/2" for x in best_offsets])
                + " ]",
            )

if __name__ == "__main__":
    joint_sign = np.array([-1, -1, -1, -1, -1, 1])

    haptics_joints = np.array([0, np.pi/2, np.pi/2, 0, 0, 0])
    # swtich the 3 and 4 elements in haptics_joints
    haptics_joints[3], haptics_joints[4] = haptics_joints[4], haptics_joints[3]
    
    robot_joints = np.array([0, -np.pi/2, np.pi/2, -np.pi/2, 0, 0])
    cur_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(cur_path, 'teleop_offset')
    
    # Calculate the offset
    offset = (haptics_joints - joint_sign*robot_joints)

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True) 

    # Use np.savez or np.savez_compressed to save multiple arrays
    file_path = os.path.join(save_dir, 'haptics_offset.npz')
    np.savez(file_path, joint_sign=joint_sign, offset=offset)


    get_haptics_offset(joint_sign)

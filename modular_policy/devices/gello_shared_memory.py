import sys
sys.path.insert(1, '.')


import multiprocessing as mp
import numpy as np
import time
import glob
from modular_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from modular_policy.common.trans_utils import matrix_to_euler
import time, os
from dataclasses import dataclass, field
from multiprocessing.managers import SharedMemoryManager
import functools
from modular_policy.shared_memory.shared_ndarray import SharedNDArray
from modular_policy.devices.gello_software.gello.agents.gello_agent import GelloAgent
from modular_policy.devices.gello_software.gello.agents.agent import BimanualAgent
import yaml

class Gello(mp.Process):

    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=200,
            max_value=500, 
            dtype=np.float32,
            verbose=False,
            use_gripper=True,
            bimanual=True,
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

        # copied variables
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        
        self.bimanual = bimanual
        if bimanual:
            # dynamixel control box port map (to distinguish left and right gello)
            gello_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config/teleop/gello.yaml")
            
            with open(gello_config_path) as stream:
                try:
                    gello_config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
            
            self.left_port = gello_config['left_gello_port']  
            self.right_port = gello_config['right_gello_port']
        else:
            self.gello_port = None
            if self.gello_port is None:
                usb_ports = glob.glob("/dev/serial/by-id/*")
                print(f"Found {len(usb_ports)} ports")
                if len(usb_ports) > 0:
                    self.gello_port = usb_ports[0]
                    print(f"using port {self.gello_port}")
                else:
                    raise ValueError(
                        "No gello port found, please specify one or plug in gello"
                    )
        self.use_griper = use_gripper
        self.num_joints = 6 + int(use_gripper)
        self.num_robots = 2 if bimanual else 1
        example = {
            'joint_state': np.zeros((self.num_joints*self.num_robots,), dtype=np.float32),
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

        

    # ======= get state APIs ==========

    def get_joint_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['joint_state'][:self.num_joints if not self.bimanual else self.num_joints*2], 
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
            print("Gello started.")
        if self.bimanual:
            left_agent = GelloAgent(port=self.left_port)
            right_agent = GelloAgent(port=self.right_port)
            agent = BimanualAgent(left_agent, right_agent, enable_gripper=self.use_griper)
        else:
            agent = GelloAgent(port=self.gello_port)

        try:
            joint_state = np.zeros((self.num_joints*self.num_robots,), dtype=np.float32)
            # send one message immediately so client can start reading
            self.ring_buffer.put({
                'joint_state': joint_state,
                'receive_timestamp': time.time()
            })
            self.ready_event.set()

            while not self.stop_event.is_set():
                if self.verbose:
                    print("Polling Gello device event...")

                receive_timestamp = time.time()
                joint_state = agent.act({})[:self.num_joints if not self.bimanual else self.num_joints*2]
                # finish integrating this round of events
                # before sending over
                self.ring_buffer.put({
                    'joint_state': joint_state,
                    'receive_timestamp': receive_timestamp
                })
                time.sleep(1/self.frequency)
        finally:
            if self.verbose:
                print("Gello process stopped.")


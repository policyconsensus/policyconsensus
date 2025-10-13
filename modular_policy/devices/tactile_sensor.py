import numpy as np
import serial
import threading
import time
from typing import Dict, Optional, Tuple


class TactileSensor:
    """Single tactile sensor interface"""
    
    def __init__(self, port: str, baud_rate: int = 2000000, threshold: int = 12, noise_scale: int = 60):
        self.port = port
        self.baud_rate = baud_rate
        self.threshold = threshold
        self.noise_scale = noise_scale
        
        # Data storage
        self.contact_data_norm = np.zeros((16, 32))
        self.is_initialized = False
        self.median_baseline = None
        
        # Threading
        self.serial_device = None
        self.read_thread = None
        self.running = False
        
    def connect(self):
        """Connect to the tactile sensor"""
        try:
            self.serial_device = serial.Serial(self.port, self.baud_rate)
            self.serial_device.flush()
            print(f"Connected to tactile sensor at {self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to tactile sensor at {self.port}: {e}")
            return False
    
    def start_reading(self):
        """Start the background reading thread"""
        if self.serial_device is None:
            raise RuntimeError("Must connect to sensor before starting reading")
        
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop)
        self.read_thread.daemon = True
        self.read_thread.start()
    
    def stop_reading(self):
        """Stop the background reading thread"""
        self.running = False
        if self.read_thread:
            self.read_thread.join()
    
    def close(self):
        """Close the serial connection"""
        self.stop_reading()
        if self.serial_device:
            self.serial_device.close()
    
    def get_contact_data(self) -> np.ndarray:
        """Get the current normalized contact data"""
        return self.contact_data_norm.copy()
    
    def is_ready(self) -> bool:
        """Check if sensor is initialized and ready"""
        return self.is_initialized
    
    def _validate_data_shape(self, data_list) -> dict:
        """Validate that all sublists have the same length to prevent inhomogeneous array issues
        
        Returns:
            dict with keys: 'valid' (bool), 'error' (str), 'expected' (int), 'actual' (list)
        """
        if not data_list:
            return {'valid': False, 'error': 'Empty data list', 'expected': 0, 'actual': []}
        
        expected_length = len(data_list[0])
        actual_lengths = [len(row) for row in data_list]
        
        if all(length == expected_length for length in actual_lengths):
            return {'valid': True, 'error': '', 'expected': expected_length, 'actual': actual_lengths}
        else:
            unique_lengths = sorted(set(actual_lengths))
            error_msg = f"Inconsistent row lengths. Expected {expected_length} values per row, found rows with {unique_lengths} values"
            return {'valid': False, 'error': error_msg, 'expected': expected_length, 'actual': actual_lengths}
    
    def _read_loop(self):
        """Main reading loop (runs in background thread)"""
        # Initialization phase
        data_tac = []
        num = 0
        current = None
        
        print(f"Initializing tactile sensor at {self.port}...")
        
        while self.running and num < 30:
            if self.serial_device.in_waiting > 0:
                try:
                    line = self.serial_device.readline().decode('utf-8').strip()
                except:
                    line = ""
                
                if len(line) < 10:
                    if current is not None and len(current) == 16:
                        # Validate that all sublists have the same length before creating array
                        validation_result = self._validate_data_shape(current)
                        if validation_result['valid']:
                            try:
                                backup = np.array(current)
                                data_tac.append(backup)
                                num += 1
                            except ValueError as e:
                                print(f"Error: Failed to create tactile array at {self.port}: {e}")
                                print(f"Data shape: {[len(row) for row in current[:3]]}...")  # Show first 3 rows
                                raise RuntimeError(f"Tactile sensor {self.port} array creation failed during initialization")
                        else:
                            print(f"Warning: Skipping tactile reading at {self.port} - {validation_result['error']}")
                            print(f"Expected length: {validation_result['expected']}, got lengths: {validation_result['actual'][:5]}...")  # Show first 5
                    current = []
                    continue
                
                if current is not None:
                    str_values = line.split()
                    try:
                        int_values = [int(val) for val in str_values]
                        current.append(int_values)
                    except ValueError as e:
                        print(f"Warning: Failed to parse tactile data line at {self.port}: '{line}' - {e}")
                        continue
        
        if len(data_tac) > 0:
            data_tac = np.array(data_tac)
            self.median_baseline = np.median(data_tac, axis=0)
            self.is_initialized = True
            print(f"Tactile sensor at {self.port} initialized with {len(data_tac)} valid readings!")
        else:
            print(f"Error: Failed to initialize tactile sensor at {self.port} - no valid data collected")
            raise RuntimeError(f"Tactile sensor {self.port} initialization failed: no valid readings obtained")
        
        # Main reading loop
        while self.running:
            if self.serial_device.in_waiting > 0:
                try:
                    line = self.serial_device.readline().decode('utf-8').strip()
                except:
                    line = ""
                
                if len(line) < 10:
                    if current is not None and len(current) == 16:
                        validation_result = self._validate_data_shape(current)
                        if validation_result['valid']:
                            try:
                                backup = np.array(current)
                                self._process_raw_data(backup)
                            except ValueError as e:
                                print(f"Error: Failed to process tactile data at {self.port}: {e}")
                                print(f"Data shape: {[len(row) for row in current[:3]]}...")  # Show first 3 rows
                                # Continue running but log the error - don't crash during normal operation
                        else:
                            print(f"Warning: Skipping tactile reading at {self.port} - {validation_result['error']}")
                            print(f"Expected length: {validation_result['expected']}, got lengths: {validation_result['actual'][:5]}...")  # Show first 5
                    current = []
                    continue
                
                if current is not None:
                    str_values = line.split()
                    try:
                        int_values = [int(val) for val in str_values]
                        current.append(int_values)
                    except ValueError as e:
                        print(f"Warning: Failed to parse tactile data line at {self.port}: '{line}' - {e}")
                        continue
    
    def _process_raw_data(self, raw_data: np.ndarray):
        """Process raw sensor data into normalized contact data"""
        if self.median_baseline is None:
            return
        
        contact_data = raw_data - self.median_baseline - self.threshold
        contact_data = np.clip(contact_data, 0, 100)
        
        if np.max(contact_data) < self.threshold:
            self.contact_data_norm = contact_data / self.noise_scale
        else:
            self.contact_data_norm = contact_data / np.max(contact_data)


class DualTactileSensor:
    """Dual tactile sensor interface for both gripper fingers"""
    
    def __init__(self, left_port: str = '/dev/ttyUSB3', right_port: str = '/dev/ttyUSB2', 
                 baud_rate: int = 2000000, threshold: int = 30, noise_scale: int = 60):
        self.left_sensor = TactileSensor(left_port, baud_rate, threshold, noise_scale)
        self.right_sensor = TactileSensor(right_port, baud_rate, threshold, noise_scale)
    
    def connect(self) -> bool:
        """Connect to both tactile sensors"""
        left_ok = self.left_sensor.connect()
        right_ok = self.right_sensor.connect()
        return left_ok and right_ok
    
    def start_reading(self):
        """Start reading from both sensors"""
        self.left_sensor.start_reading()
        self.right_sensor.start_reading()
    
    def stop_reading(self):
        """Stop reading from both sensors"""
        self.left_sensor.stop_reading()
        self.right_sensor.stop_reading()
    
    def close(self):
        """Close both sensor connections"""
        self.left_sensor.close()
        self.right_sensor.close()
    
    def get_contact_data(self) -> Dict[str, np.ndarray]:
        """Get contact data from both sensors"""
        return {
            'left_tactile': self.left_sensor.get_contact_data(),
            'right_tactile': self.right_sensor.get_contact_data()
        }
    
    def is_ready(self) -> bool:
        """Check if both sensors are ready"""
        return self.left_sensor.is_ready() and self.right_sensor.is_ready()
    
    def __enter__(self):
        self.connect()
        self.start_reading()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
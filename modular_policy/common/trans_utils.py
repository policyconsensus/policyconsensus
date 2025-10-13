import numpy as np
from math import sqrt, atan2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d

def inverse_transform(transformation_matrix):
    """
    Computes the inverse of a transformation matrix.

    Parameters:
    -----------
    old_transform : np.ndarray
        The original transformation matrix.

    Returns:
    --------
    np.ndarray
        The inverse transformation matrix.
    """
    R = transformation_matrix[:3, :3]
    t = transformation_matrix[:3, 3]
    
    # For an orthogonal matrix, the inverse is equal to its transpose
    R_inv = R.T  # Transpose of the rotation matrix

    # Compute the inverse translation vector
    t_inv = -R_inv @ t  # Using matrix multiplication for the inverse translation

    # Construct the inverse transformation matrix
    inverse_matrix = np.eye(4)
    inverse_matrix[:3, :3] = R_inv
    inverse_matrix[:3, 3] = t_inv

    return inverse_matrix


def rotation_matrix_zaxis(degrees):
    radians = np.deg2rad(degrees)
    cos_val, sin_val = np.cos(radians), np.sin(radians)
    return np.array([[cos_val, -sin_val, 0, 0],
                     [sin_val, cos_val, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])



def remove_scaling_from_transformation_matrix(T):
    # Extract the rotation part
    R = T[:3, :3]
    
    # Normalize each column vector using vectorized operations
    norms = np.linalg.norm(R, axis=0)
    R_normalized = R / norms

    # Reassemble the transformation matrix without scaling
    T_no_scale = T.copy()
    T_no_scale[:3, :3] = R_normalized
    return T_no_scale



def apply_constraints_to_transformation(transformation_matrix, fixed_roll=180, fixed_pitch=0, fixed_z=0.0455):
    # Extract yaw and compute rotation matrix efficiently
    yaw = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0])
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    transformation_matrix[:3, :3] = [[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]]
    transformation_matrix[2, 3] = fixed_z

    return transformation_matrix

def get_angle_from_rotation_matrix(R):
    """
    Calculates the angle of rotation from a given rotation matrix.

    Args:
        R (numpy.ndarray): The rotation matrix.

    Returns:
        float: The angle of rotation in degrees.
    """
    # https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    sy = sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = atan2(R[2, 1], R[2, 2])
        y = atan2(-R[2, 0], sy)
        z = atan2(R[1, 0], R[0, 0])
    else:
        x = atan2(-R[1, 2], R[1, 1])
        y = atan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])


def compute_center_in_world(trans_local_world, local_center_homogeneous):
    """Computes the center of the object in world coordinates."""
    return np.dot(trans_local_world, local_center_homogeneous)

def compute_center_in_simulation(obj_center_world, trans_real_in_sim):
    """Transforms the center from world coordinates to simulation coordinates."""
    obj_center_world_2d = obj_center_world[:2]
    obj_center_world_homogeneous = np.hstack((obj_center_world_2d, np.array([1])))
    return np.dot(trans_real_in_sim, obj_center_world_homogeneous.T).T

def extract_yaw_angle(rotation_matrix):
    """Extracts the yaw angle from a rotation matrix."""
    return np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

def compute_yaw_in_simulation(yaw_local_world, trans_real_in_sim):
    """Computes the yaw rotation in simulation coordinates."""
    yaw_world_sim = extract_yaw_angle(trans_real_in_sim[:2, :2])
    return yaw_local_world + yaw_world_sim



# Corrected transformation functions using 4x4 matrices
def transform_to_world(robot_base_pose, position):
    """
    Transforms a position from the robot's base frame to the world frame using a 4x4 transformation matrix.

    Args:
    - robot_base_pose (np.ndarray): The robot's base pose in the world as a 4x4 transformation matrix.
    - position (np.ndarray): The position in the robot's base frame to be transformed.

    Returns:
    - np.ndarray: The transformed position in the world frame.
    """
    # Create a homogeneous coordinate representation of the position
    position_hom = np.append(position, 1)
    
    # Apply the transformation matrix
    position_world_hom = np.dot(robot_base_pose, position_hom)
    
    # Convert back from homogeneous coordinates
    position_world = position_world_hom[:3]
    return position_world

def transform_from_world(robot_base_pose, position_world):
    """
    Transforms a position from the world frame back to the robot's base frame using the inverse of a 4x4 transformation matrix.

    Args:
    - robot_base_pose (np.ndarray): The robot's base pose in the world as a 4x4 transformation matrix.
    - position_world (np.ndarray): The position in the world frame to be transformed back.

    Returns:
    - np.ndarray: The transformed position in the robot's base frame.
    """
    # Invert the transformation matrix to go from world to robot base frame
    robot_base_pose_inv = np.linalg.inv(robot_base_pose)
    
    # Create a homogeneous coordinate representation of the world position
    position_world_hom = np.append(position_world, 1)
    
    # Apply the inverse transformation matrix
    position_base_hom = np.dot(robot_base_pose_inv, position_world_hom)
    
    # Convert back from homogeneous coordinates
    position_base = position_base_hom[:3]
    return position_base


def matrix_to_euler(R):
    """
    Convert a rotation matrix to Euler angles.
    """
    sy = sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])

    singular = sy < 1e-6

    if not singular:
        x = atan2(R[2][1], R[2][2])
        y = atan2(-R[2][0], sy)
        z = atan2(R[1][0], R[0][0])
    else:
        x = atan2(-R[1][2], R[1][1])
        y = atan2(-R[2][0], sy)
        z = 0

    return [x, y, z]



def interpolate_poses(start_pose, final_pose, speed, translation_tolerance=0.02, rotation_tolerance=np.deg2rad(5)):
    # Split the start and final poses into translation and rotation components
    if len(start_pose) != len(final_pose):
        raise ValueError("Start and final poses must have the same length.")
    
    start_translation, start_rotation = start_pose[:3], start_pose[3:6]
    final_translation, final_rotation = final_pose[:3], final_pose[3:6]
    
    # Calculate the total distance for translation
    translation_distance = np.linalg.norm(final_translation - start_translation)
    
    # Calculate rotation angle difference using quaternion distance
    start_quat = R.from_rotvec(start_rotation).as_quat()
    final_quat = R.from_rotvec(final_rotation).as_quat()
    rotation_angle_difference = R.from_quat([start_quat]).inv() * R.from_quat([final_quat])
    rotation_angle_difference = rotation_angle_difference.magnitude()
    
    # Check if start and final poses are close
    if translation_distance <= translation_tolerance and rotation_angle_difference <= rotation_tolerance:
        # Poses are considered close, return an indication and skip interpolation
        return [], 0, True
    
    # Calculate the number of steps required based on the given speed
    total_steps = max(int(np.ceil(translation_distance / speed)), 1)
    
    # Setup for Slerp and linear interpolation
    times = [0, 1]  # Start and end times
    key_rotations = R.from_quat([start_quat, final_quat])
    slerp = Slerp(times, key_rotations)
    proportional_times = np.linspace(0, 1, total_steps + 2)[1:-1]
    interp_func = interp1d([0, 1], np.vstack([start_translation, final_translation]), axis=0)
    
    # Perform interpolations
    interpolated_rotations = slerp(proportional_times)
    interpolated_translations = interp_func(proportional_times)
    
    interpolated_poses = []

    if len(start_pose) == 7 and len(final_pose) == 7:
        # If 7D poses, assume the last element is a gripper state
        final_gripper_pose = final_pose[6]
        # Combine interpolated translations, rotations, and gripper state
        for i in range(total_steps):
            pose = np.hstack([interpolated_translations[i], interpolated_rotations[i].as_rotvec(), final_gripper_pose])
            interpolated_poses.append(pose)
    else:
        # For 6D poses, just combine translations and rotations
        for i in range(total_steps):
            pose = np.hstack([interpolated_translations[i], interpolated_rotations[i].as_rotvec()])
            interpolated_poses.append(pose)
            
    return interpolated_poses, total_steps, False

def interpolate_poses_12d(start_pose, final_pose, speed, translation_tolerance=0.02, rotation_tolerance=np.deg2rad(5)):
    """
    Interpolates between two 12D poses, where each 6D subset represents a separate robot.
    """
    # Split into two 6D poses (first and second robots)
    start_pose_robot1, start_pose_robot2 = start_pose[:6], start_pose[6:]
    final_pose_robot1, final_pose_robot2 = final_pose[:6], final_pose[6:]
    
    # Interpolate for both robots separately using the existing 6D interpolate_poses function
    interpolated_robot1, steps1, close1 = interpolate_poses(
        start_pose_robot1, final_pose_robot1, speed, translation_tolerance, rotation_tolerance)
    
    interpolated_robot2, steps2, close2 = interpolate_poses(
        start_pose_robot2, final_pose_robot2, speed, translation_tolerance, rotation_tolerance)
    
    # Check if both robots are already close enough
    if close1 and close2:
        return [], 0, True
    
    # Determine the maximum number of steps between both robots
    max_steps = max(steps1, steps2)
    
    # Expand to ensure both have the same number of interpolated steps
    interpolated_robot1 += [final_pose_robot1] * (max_steps - len(interpolated_robot1))
    interpolated_robot2 += [final_pose_robot2] * (max_steps - len(interpolated_robot2))
    
    # Combine both sets to create a full 12D pose
    interpolated_poses = [
        np.hstack([robot1_pose, robot2_pose])
        for robot1_pose, robot2_pose in zip(interpolated_robot1, interpolated_robot2)
    ]
    
    return interpolated_poses, max_steps, False

def interpolate_joints(start_joints, final_joints, speed=0.1, num_steps=None):
    """
    Interpolates joint values linearly from start to final configurations.

    Parameters:
    - start_joints: Array of starting joint values.
    - final_joints: Array of final joint values.
    - speed: Approximate step size for interpolation, not used if num_steps is provided.
    - num_steps: Exact number of interpolation steps to use.

    Returns:
    - interpolated_joints: Array of interpolated joint configurations.
    """
    # Calculate the max joint difference to estimate steps if num_steps is not specified
    if num_steps is None or num_steps < 1:
        max_delta = np.max(np.abs(final_joints - start_joints))
        num_steps = max(int(np.ceil(max_delta / speed)), 1)
    
    # Generate interpolation steps
    step_ratios = np.linspace(0, 1, num_steps + 2)[1:-1]  # Exclude the start and end point for interpolation
    
    # Prepare the array for interpolated joints
    interpolated_joints = []
    for ratio in step_ratios:
        # Linear interpolation
        interpolated_config = (1 - ratio) * start_joints + ratio * final_joints
        interpolated_joints.append(interpolated_config)
    
    return np.array(interpolated_joints)

def normalize_angles(angles):
    """
    Normalize angles to be within the [0, 2π] range.
    """
    return np.mod(angles, 2 * np.pi)

def are_joints_close(joint_vectors_1, joint_vectors_2, threshold=np.pi/180):
    """
    Determine if pairs of robot joint angle vectors are close enough based on a threshold,
    considering the 2π periodicity of angles. This function is vectorized for efficiency.

    Parameters:
        joint_vectors_1 (np.ndarray): First array of joint angle vectors in radians (N x 6).
        joint_vectors_2 (np.ndarray): Second array of joint angle vectors in radians (N x 6).
        threshold (float): The maximum Euclidean distance between vectors to be considered close.

    Returns:
        np.ndarray: A boolean array of length N, where True indicates the vectors are close.
    """
    # Normalize angles to [0, 2π] for a proper comparison
    joint_vectors_1_normalized = normalize_angles(joint_vectors_1)
    joint_vectors_2_normalized = normalize_angles(joint_vectors_2)

    # Calculate the Euclidean distances between the pairs of normalized vectors
    distances = np.linalg.norm(joint_vectors_1_normalized - joint_vectors_2_normalized, axis=1)
    
    # Determine if the distances are within the threshold
    return distances <= threshold


def convert_position_rotation_to_homogeneous(input_vector):
    """
    Convert a 3D transformation (position and rotation vector) to a homogeneous transformation matrix.
    
    Parameters:
    input_vector (list or array): A list or array of 6 elements containing the position (x, y, z)
                                  and the rotation vector (rx, ry, rz).
    
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix.
    """
    position = np.array(input_vector[:3])
    rotation_vector = np.array(input_vector[3:])
    rotation = R.from_rotvec(rotation_vector)
    rotation_matrix = rotation.as_matrix()
    homogeneous_matrix = np.eye(4)
    homogeneous_matrix[:3, :3] = rotation_matrix
    homogeneous_matrix[:3, 3] = position
    return homogeneous_matrix

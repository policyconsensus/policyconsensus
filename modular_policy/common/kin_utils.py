import numpy as np
import open3d as o3d
from klampt import WorldModel
from klampt.io import numpy_convert, open3d_convert
from klampt.math import vectorops,so3,se3
import os 
import scipy.spatial
from scipy.spatial.transform import Rotation as R
import sys
sys.path.insert(1, '.')
from modular_policy.common.trans_utils import convert_position_rotation_to_homogeneous

# from modular_policy.common.pcd_utils import visualize_pcd
from numba import jit


def teleop_to_robot_joints(joint_sign, device_state, offset):
    return joint_sign* (device_state - offset)



class RobotKinHelper:
    def __init__(self, robot_name, tool_name=None, N_per_link=1000, N_tool=100, debug=False):
        assert robot_name == 'ur5e', 'Only ur5e is supported'
        o3d.utility.random.seed(0)

        # Load the robot model
        real_world_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'real_world')
        self.robot_name = robot_name
        robot_model_path = os.path.join(real_world_path, 'ur5e_hires.rob')
        self.robot_model_path = robot_model_path
        self.world = WorldModel()
        success = self.world.readFile(robot_model_path)
        if not success:
            raise IOError("Failed to load robot model from path: {}".format(robot_model_path))
        self.robot_model = self.world.robot(0)
        self.debug = debug
        self.ee_additional_rotation = R.from_euler('x', -np.pi / 2, degrees=False).as_matrix()
        
        self.base_joint_offset = [np.pi]
        self.ee_joint_offset = [np.pi/2*3]        
        self.mesh_cache = []
        self.link_pcds = []

        self.num_links = self.robot_model.numLinks()

        self.initialize_mesh_cache(N_per_link, robot_name)
        
        self.tool_name = tool_name
        self.N_tool = N_tool
        self.hande_range = 0.05
        if tool_name == 'knife':
            self.tool_pcd = self.load_knife_pcd(N_tool)
        elif tool_name == 'bare_gripper' :
            self.tool_pcd = self.load_bare_gripper_pcd(N_tool)
        elif tool_name == 'soft_gripper':
            self.tool_pcd = self.load_soft_gripper_pcd(N_tool)
        elif tool_name == 'soft_hande':
            # concatenate the point cloud from soft_gripper and bare_gripper
            soft_fingers_pcd = self.load_soft_gripper_pcd(int(N_tool/2))
            points = np.asarray(soft_fingers_pcd.points)
            points[:, 2] += 0.115
            soft_fingers_pcd.points = o3d.utility.Vector3dVector(points)
            
            bare_gripper_pcd = self.load_bare_gripper_pcd(int(N_tool/2))
            self.tool_pcd = soft_fingers_pcd + bare_gripper_pcd

        else:
            self.tool_pcd = None

        if debug and self.tool_pcd is not None:
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([self.tool_pcd, coordinate_frame])

        self.tcp_offsets = {
            'knife': np.array([0.03, 0.04, 0.11, 0, 0, np.pi]),
            'bare_gripper': np.array([0.0, 0.0, 0.0, 0, 0, 0]),
            'soft_gripper': np.array([0.0, 0.0, 0.115, 0, 0, 0]),
            'soft_hande': np.array([0.0, 0.0, 0.0, 0, 0, 0]),

        }

        # Precompute transformation matrices for TCP offsets if they are static
        def euler_vector_to_matrix(tcp_offset_pose_vector):
            tcp_transform = np.eye(4)
            tcp_transform[:3, 3] = tcp_offset_pose_vector[:3]
            tcp_transform[:3, :3] = R.from_euler('xyz', tcp_offset_pose_vector[3:], degrees=False).as_matrix()
            return tcp_transform
        
        
        self.tcp_transforms = {
            tool: euler_vector_to_matrix(offset)
            for tool, offset in self.tcp_offsets.items()
        }

    def initialize_mesh_cache(self, N_per_link, robot_name):
        """Cache meshes and precompute normals for each link."""
        for link_idx in range(self.num_links):
            mesh = open3d_convert.to_open3d(self.robot_model.link(link_idx).geometry().getTriangleMesh())
            mesh.compute_vertex_normals()
            # mesh.compute_triangle_normals()
            self.mesh_cache.append(mesh)
            pcd = mesh.sample_points_poisson_disk(number_of_points=N_per_link)
            self.link_pcds.append(pcd)
            
    def load_knife_pcd(self, N_tool=100):
        assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
        tool_path = os.path.join(assets_path, 'tools/knife_simple.STL')
        tool_mesh = o3d.io.read_triangle_mesh(tool_path)
        tool_pcd = tool_mesh.sample_points_poisson_disk(number_of_points=N_tool)
        points = np.asarray(tool_pcd.points)/1000
        tool_pcd.points = o3d.utility.Vector3dVector(points)
        return tool_pcd

    def load_bare_gripper_pcd(self, N_tool=100):
        assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
        tool_path = os.path.join(assets_path, 'tools/hand-e-collision.stl')
        tool_mesh = o3d.io.read_triangle_mesh(tool_path)
        tool_pcd = tool_mesh.sample_points_poisson_disk(number_of_points=N_tool)
        return tool_pcd

    def load_soft_gripper_pcd(self, N_tool=100):
        assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')
        tool_path = os.path.join(assets_path, 'tools/soft_gripper.STL')
        tool_mesh = o3d.io.read_triangle_mesh(tool_path)
        tool_pcd = tool_mesh.sample_points_poisson_disk(number_of_points=N_tool)
        points = np.asarray(tool_pcd.points)/1000
        points[:, 0] -= 0.127

        points_copy1 = np.copy(points)
        points_copy2 = np.copy(points)
        # Define and apply the rotation matrix for a 180-degree rotation around the x-axis
        rotation_matrix_x = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        points_copy1 = points_copy1.dot(rotation_matrix_x)
        points_copy1[:, 1] += 0.023
        points_copy1[:, 2] += (0.04*2+0.05)
        rotation_matrix_y = np.array([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ])

        tool_points = np.vstack([points_copy1, points_copy2])
        tool_points[:, 2] -= (0.04+0.025)
        tool_points = tool_points.dot(rotation_matrix_y)
        tool_points[:, 1] -= 0.023/2

        tool_pcd.points = o3d.utility.Vector3dVector(tool_points)
        return tool_pcd
    
    def get_forward_kin(self, robot_q, tcp_offset_pose_vector, out_homo=False):
        """
        Computes the tool pose given the joint values and TCP offset pose.
        
        Parameters:
        - joint_values: A list or numpy array of joint values for the robot.
        - tcp_offset_pose_vector: A list or numpy array representing the TCP offset pose in the format
                                  [X, Y, Z, Rx, Ry, Rz], where X, Y, Z are translation components, and
                                  Rx, Ry, Rz are rotation components in radians.
        
        Returns:
        - A list representing the tool pose in the format [X, Y, Z, Rx, Ry, Rz], where X, Y, Z are the
          translation components, and Rx, Ry, Rz are the rotation components in radians.
        """
        robot_q = np.array(robot_q)
        use_gripper = len(robot_q) == 7

        if use_gripper:
            gripper_q = robot_q[-1]
            robot_q = robot_q[:-1]

        robot_q = np.concatenate([self.base_joint_offset, robot_q, self.ee_joint_offset])

        # Set the robot configuration
        self.robot_model.setConfig(robot_q.tolist())
        
        # Compute the forward kinematics to get the end-effector pose
        ee_transform = numpy_convert.to_numpy(self.robot_model.link(self.robot_model.numLinks() - 1).getTransform())
        ee_transform[:3, :3] = np.dot(ee_transform[:3, :3], self.ee_additional_rotation) 

        # Final transformation
        tool_transform = np.dot(ee_transform, self.tcp_transforms[self.tool_name])
        if out_homo:
            # Return the full 4x4 homogeneous transformation matrix
            return tool_transform
        else:
            # Extract translation and rotation vector
            tool_translation = tool_transform[:3, 3]
            tool_rotation_vec = R.from_matrix(tool_transform[:3, :3]).as_rotvec()
            tool_pose_vector = np.concatenate((tool_translation, tool_rotation_vec))
            if use_gripper:
                tool_pose_vector = np.append(tool_pose_vector, gripper_q)
            return tool_pose_vector.tolist()

        
        # tool_translation = tool_transform[:3, 3]
        # tool_rotation_vec = R.from_matrix(tool_transform[:3, :3]).as_rotvec()

        # tool_pose_vector = np.concatenate((tool_translation, tool_rotation_vec))
        # if use_gripper:
        #     tool_pose_vector = np.append(tool_pose_vector, gripper_q)

        # return tool_pose_vector.tolist()

    def get_robot_pcd(self, qpos, N_joints=None, N_per_link=200, surface_only=False, out_o3d=True):
        robot_q = np.concatenate([self.base_joint_offset, np.asarray(qpos, dtype=np.float64), self.ee_joint_offset])
        self.robot_model.setConfig(robot_q.tolist())

        base_pcd = o3d.geometry.PointCloud()
        # Determine the starting link index based on N_joints
        start_link_idx = 0 if N_joints is None else max(0, self.robot_model.numLinks() - N_joints)

        for link_idx in range(start_link_idx, self.num_links):
            mesh = self.mesh_cache[link_idx]
            link_transform = numpy_convert.to_numpy(self.robot_model.link(link_idx).getTransform())

            if surface_only:
                sampled_pcd = mesh.sample_points_uniformly(number_of_points=N_per_link)
            else:
                sampled_pcd = self.link_pcds[link_idx]
    
            # Apply transformations
            points = np.asarray(sampled_pcd.points)
            points_hom = np.hstack([points, np.ones((points.shape[0], 1))])  # Add homogeneous coordinates
            trans_points = (link_transform @ points_hom.T)[:3, :].T

            # Create transformed point cloud
            trans_pcd = o3d.geometry.PointCloud()
            trans_pcd.points = o3d.utility.Vector3dVector(trans_points)
            base_pcd += trans_pcd

        return base_pcd if out_o3d else np.asarray(base_pcd.points)


    def visualize_robot_pointcloud(self, robot_q):
        """
        Visualizes the robot's point cloud based on the given configuration.
        
        Parameters:
        - robot_q (list or np.ndarray): The robot's configuration as a list or numpy array.
        """
        robot_pcd = self.get_robot_pcd(robot_q, out_o3d=True)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        
        # Compute the forward kinematics to get the end-effector pose
        ee_link = self.robot_model.link(self.robot_model.numLinks() - 1)  # Assuming the last link is the end-effector
        R_ee, t_ee = ee_link.getTransform()
        
        ee_transform = numpy_convert.to_numpy((R_ee, t_ee))
        # Combine the additional rotation with the end-effector's transformation
        ee_transform[:3, :3] = np.dot(self.ee_additional_rotation, ee_transform[:3, :3])

        # Create the end-effector's coordinate frame and apply the transformation
        ee_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        ee_coordinate_frame.transform(ee_transform)
        
        o3d.visualization.draw_geometries([robot_pcd, coordinate_frame, ee_coordinate_frame])
        
    def get_tool_pcd(self, joint_values, tcp_offset_pose_vector=None, N_per_inst=300, out_o3d=False):
        """
        Computes the tool pose given the joint values and TCP offset pose.
        
        Parameters:
        - joint_values: A list or numpy array of joint values for the robot.
        - tcp_offset_pose_vector: A list or numpy array representing the TCP offset pose in the format
                                  [X, Y, Z, Rx, Ry, Rz], where X, Y, Z are translation components, and
                                  Rx, Ry, Rz are rotation components in radians.
        
        Returns:
        - A list representing the tool pose in the format [X, Y, Z, Rx, Ry, Rz], where X, Y, Z are the
          translation components, and Rx, Ry, Rz are the rotation components in radians.
        """
        tool_name = self.tool_name 
        if tool_name not in self.tcp_transforms:
            raise RuntimeError("Unsupported tool: ", tool_name)
        
        gripper_value = 0
        if len(joint_values) == 7:
            gripper_value = joint_values[-1]
            joint_values = joint_values[:6]
            
        tcp_offset_pose_vector = self.tcp_offsets[tool_name]
        tool_pose_homo = self.get_forward_kin(joint_values, tcp_offset_pose_vector, out_homo=True)

        points = np.asarray(self.tool_pcd.points).copy()
        if tool_name == 'soft_hande':
            N_per_finger = int(self.N_tool/2)
            
            points[:N_per_finger, 0] -= self.hande_range/2*gripper_value
            points[N_per_finger:int(N_per_finger*2), 0] += self.hande_range/2*gripper_value

        points_hom = np.hstack([points, np.ones((points.shape[0], 1))])  # Add homogeneous coordinates
        trans_points = (tool_pose_homo @ points_hom.T)[:3, :].T

        # Create transformed point cloud
        trans_pcd = o3d.geometry.PointCloud()
        trans_pcd.points = o3d.utility.Vector3dVector(trans_points)

        return trans_pcd if out_o3d else np.asarray(trans_pcd.points)

        # return transformed_pcd if out_o3d else np.asarray(transformed_pcd.points)


if __name__ == "__main__":
    calculator = RobotKinHelper(robot_name = 'ur5e', )
    joint_values = np.array([np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0])  # Example joint values
    
    tool_pose_vector = calculator.compute_tool_pcd(joint_values, tool_name='knife')
    print("Tool Pose Vector:", tool_pose_vector)

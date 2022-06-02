import os
import numpy as np
from scipy.spatial.transform import Rotation as R

DEVICE = "cuda:0"
EPSILON = 1e-6

class Config:
    def __init__(self):
        self.package_name = "psdf"
        # self.path = "/home/amax_djh/catkin_ws/src/" + self.package_name

        # self.output_path = os.path.join(self.path, "output")
        # if not os.path.exists(self.output_path):
        #     os.mkdir(self.output_path)

        # vaccum cup
        self.gripper_radius = 0.01
        self.gripper_height = 0.02
        self.gripper_vertices = 8
        self.gripper_angle_threshold = 45
        self.vacuum_length = 0.125

        # psdf range
        # self.x_min = 0.48/2 + 0.1
        # self.x_max = self.x_min + 0.32
        # self.y_min = -0.58/2 +0.12
        # self.y_max = self.y_min + 0.32
        # self.z_min = 0.0223
        # self.z_max = self.z_min + 0.2
        self.x_min = -0.25
        self.x_max = self.x_min + 0.5
        self.y_min = -0.25
        self.y_max = self.y_min + 0.5
        self.z_min = 0.0
        self.z_max = self.z_min + 0.5
        self.lower_bound = np.array([self.x_min, self.y_min, self.z_min])
        self.upper_bound = np.array([self.x_max, self.y_max, self.z_max])
        
        # voxel resolution
        self.resolution = 0.002

        # setting init pose
        # which camera is on top of the middle of workspace
        self.init_cam_height = 0.35
        mid_point = (self.upper_bound + self.lower_bound) / 2
        T_cam2world = np.eye(4)
        T_cam2world[:3, :3] = R.from_euler("xyz", [180, 0.1, -90], degrees=True).as_matrix()# + np.eye(3) * 1e-3
        # print(np.linalg.det(T_cam2world[:3, :3]))
        T_cam2world[:3, 3] = mid_point
        T_cam2world[2, 3] = self.init_cam_height
        T_cam2tool0 = np.loadtxt(
            os.path.join(os.path.dirname(__file__), "../config/camera_pose_base_tip.txt"))
        T_tool02world = T_cam2world @ np.linalg.inv(T_cam2tool0)
        self.init_position = T_tool02world[:3, 3]
        self.init_quaternion = R.from_matrix(T_tool02world[:3, :3]).as_quat()
        self.init_pose = self.init_position.tolist() + self.init_quaternion.tolist()



config = Config()
from torch.multiprocessing import Lock, Process, set_start_method

from realsense_commander import RealSenseCommander

try:
     set_start_method('spawn')
except RuntimeError:
    pass

import numpy as np
from perception import DepthImage, CameraIntrinsics
from scipy.spatial.transform import Rotation as R
import torch
import cv2
import time

from scripts.psdf import PSDF
from scripts.utils import compute_surface_normal_torch


class PSDFCommander(object):
    def __init__(self, arm, bounds, resolution, device="cpu"):
        self.arm = arm
        self.resolution = resolution
        self.device = device

        self.psdf = PSDF(bounds, resolution=self.resolution, device=self.device, with_color=True)
        self.psdf_lock = Lock()
        self.integrate_process = Process(target=self.integrate)
        self.stop = False
        self.integrate_process.start()
        print("psdf start integrating")

    def get_cur_pose(self, T_cam2tool0):
        tool0_pose = self.arm.get_pose()
        T_tool02world = np.eye(4)
        T_tool02world[:3, :3] = R.from_quat(tool0_pose[3:]).as_matrix()
        T_tool02world[:3, 3] = tool0_pose[:3]
        T_cam2world = T_tool02world @ T_cam2tool0
        return T_cam2world

    def integrate(self):
        # cannot pickle pyrealsense2 object
        cam = RealSenseCommander()
        cam.start()
        T_cam2tool0 = np.loadtxt("config/camera_pose_base_tip.txt")
        cam_intr = np.array(CameraIntrinsics.load("config/realsense.intr").K)

        while not self.stop:
            color, depth = cam.get_image()
            T_cam2world = self.get_cur_pose(T_cam2tool0)
            depth = DepthImage(depth).inpaint(0.5).data

            self.psdf_lock.acquire()
            self.psdf.psdf_integrate(depth, cam_intr, T_cam2world, color=color)
            self.psdf_lock.release()
            # print("integrated")

    def close(self):
        self.stop = True
        self.integrate_process.join()

    def get_data(self, smooth=False, ksize=5, sigmaColor=0.1, sigmaSpace=5):
        # return None, None, None, None

        self.psdf_lock.acquire()

        # get height map
        height, width, _ = self.psdf.sdf_volume.shape
        mask = self.psdf.sdf_volume <= 0.01
        mask_flat = torch.max(mask, dim=-1)[0]
        z_vol = torch.zeros_like(self.psdf.sdf_volume).long()
        z_vol[mask] = self.psdf.voxel_coordinates[mask][:, 2]
        z_flat = torch.max(z_vol, dim=-1)[0]
        height_map = self.psdf.world_coordinates[..., 2].take(z_flat)
        if smooth:
            height_map = cv2.bilateralFilter(height_map, ksize, sigmaColor, sigmaSpace)

        # get variance
        variances = self.psdf.variance_volume.take(z_flat)
        variances[~mask_flat] = 10

        # get point map
        points = self.psdf.world_coordinates[:, :, 0, :].clone()
        points[..., 2] = height_map

        # get color map
        colors = self.psdf.decode_color(self.psdf.color_volume.take(z_flat))

        # re-arrangement
        variances = torch.flip(variances, dims=[0,1])
        points = torch.flip(points, dims=[0,1])
        # normals = torch.flip(normals, dims=[0,1])
        colors = torch.flip(colors, dims=[0,1])

        # get normal map
        normals = compute_surface_normal_torch(points)

        self.psdf_lock.release()
        return points, normals, variances, colors

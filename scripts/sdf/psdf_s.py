import torch
import numpy as np
import math
from .psdf import PSDF

class PSDFs(PSDF):
    def __init__(self, volume_bounds, resolution, device="cuda:0", with_color=False):
        super(PSDFs, self).__init__(volume_bounds, resolution, device, with_color)

    def psdf_integrate(self, depth, intrinsic, camera_pose, color=None):
        """
        Integrate an RGB-D frame into SDF volume
        :param color: A HxWx3 numpy array representing a color image
        :param depth: A HxW numpy array representing a depth map
        :param intrinsic: A 3x3 numpy array representing camera intrinsic matrix
        :param camera_pose: A 4x4 transformation matrix representing the pose from world to camera
        """
        height, width = depth.shape
        if self.with_color and color is not None:
            color = self.encode_color(torch.FloatTensor(color).to(self.device))
        depth = torch.FloatTensor(depth).to(self.device)

        T_cam2world = torch.FloatTensor(camera_pose).to(self.device)
        T_world2cam = torch.inverse(T_cam2world).to(self.device)
        T_cam2img = torch.FloatTensor(intrinsic).to(self.device)

        # find all voxel within the camera view
        cam_coors = self.rigid_transform(self.world_coordinates, T_world2cam)
        img_coors = self.camera2pixel(cam_coors, T_cam2img)
        valid_mask = (
                (0 <= img_coors[..., 0]) * (img_coors[..., 0] < width)
                * (0 <= img_coors[..., 1]) * (img_coors[..., 1] < height)
                * (cam_coors[..., 2] > 0)
        ).bool()
        voxel_coors = self.voxel_coordinates[valid_mask].long()
        pixel_coors = img_coors[valid_mask].long()
        x, y, z = voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2]
        v, u = pixel_coors[:, 0], pixel_coors[:, 1]

        # get truncated distance
        volume_depth = cam_coors[x, y, z, 2]
        surface_depth = depth[u, v]
        if self.with_color and color is not None:
            surface_color = color[u, v]
        distance = surface_depth - volume_depth
        dist_filter = (
                (surface_depth > 0)
                * (distance >= -self.truncate_margin)
        ).bool()
        x, y, z = x[dist_filter], y[dist_filter], z[dist_filter]
        distance = distance[dist_filter]
        surface_depth = surface_depth[dist_filter]
        if self.with_color and color is not None:
            surface_color = surface_color[dist_filter]
        dist_trunc = torch.clamp_max(distance/self.truncate_margin, 1)


        # update volume
        var_sensor = self.axial_noise_model(surface_depth) ** 2
        var_old = self.variance_volume[x, y, z]
        sdf_old = self.sdf_volume[x, y, z]
        p = var_sensor / (var_sensor + var_old)
        q = 1 - p
        self.sdf_volume[x, y, z] = p * sdf_old + q * dist_trunc
        # dynamic-sensitive update variance
        sign = ((sdf_old * dist_trunc) < 0).int()
        alpha = 1e-2
        lambd = 0
        diff = torch.abs(sdf_old - dist_trunc)
        var_c = sign * (torch.exp(alpha * diff)-1) + (1-sign) * lambd
        self.variance_volume[x, y, z] = p * var_old + var_c

        if self.with_color and color is not None:
            color_old = self.decode_color(self.color_volume[x, y, z])
            self.color_volume[x, y, z] = self.encode_color(
                p[..., None] * color_old + q[..., None] * self.decode_color(surface_color)
            )
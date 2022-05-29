import torch
import numpy as np
import math


class PSDF:
    def __init__(self, volume_bounds, resolution, device="cuda:0", with_color=False):
        self.device = device

        # Volume parameters
        self.volume_bounds = volume_bounds
        self.volume_bounds[:, 0] -= 20*resolution
        self.volume_bounds[:, 1] += 20*resolution
        self.resolution = resolution
        self.truncate_margin = resolution * 5
        # Adjust volume bounds
        self.volume_dims = np.ceil((self.volume_bounds[:, 1]-self.volume_bounds[:, 0])/self.resolution).astype(np.int) + 1
        self.volume_bounds[:, 1] = self.volume_bounds[:, 0] + (self.volume_dims - 1) * self.resolution
        print("boundary", self.volume_bounds)
        print("index", self.volume_dims)
        self.volume_bounds = torch.FloatTensor(self.volume_bounds).to(device)

        # Initialize volume and weights
        self.sdf_volume = torch.ones(self.volume_dims.tolist()).float().to(device)
        # self.color_volume = torch.zeros(self.volume_dims.tolist()).float().transpose(0, 1)

        xv, yv, zv = torch.meshgrid(torch.arange(self.volume_dims[0]),
                                    torch.arange(self.volume_dims[1]),
                                    torch.arange(self.volume_dims[2])) # 'ijk' indexing, i as x-axis, j as y-axis
        self.voxel_coordinates = torch.stack([xv, yv, zv], dim=-1).to(device)
        self.world_coordinates = self.volume_bounds[:, 0].reshape(1,3) + self.voxel_coordinates.float() * self.resolution
        self.variance_volume = torch.ones(self.volume_dims.tolist()).float().to(device) * 1e-3

        self.volume_dims = torch.IntTensor(self.volume_dims).to(device)

        self.with_color = with_color
        if with_color:
            self.color_volume = torch.zeros_like(self.sdf_volume)

    def decode_color(self, rgb_values):
        b = torch.floor(rgb_values / (256 * 256))
        g = torch.floor((rgb_values - b * (256 * 256)) / 256)
        r = torch.floor(rgb_values - b * (256 * 256) - g * 256)
        color = torch.stack([r, g, b], dim=-1)
        return color

    def encode_color(self, color):
        color = color.float()
        return color[..., 2] * 256 * 256 + color[..., 1] * 256 + color[..., 0]

    def camera2pixel(self, points, intrinsic):
        pixel_coordinates = torch.round((points / points[..., 2:3]) @ intrinsic.T)
        return pixel_coordinates[..., 0:2].long()

    def rigid_transform(self, points, pose):
        homogeneous_points = torch.cat([points, torch.ones((*points.shape[:-1], 1), dtype=points.dtype).to(points.device)], dim=-1)
        transformed_points = homogeneous_points @ pose.T
        return transformed_points[..., :3]

    def psdf_integrate(self, depth, intrinsic, camera_pose, color=None):
        """
        Integrate an RGB-D frame into SDF volume
        :param color: A HxWx3 numpy array representing a color image
        :param depth: A HxW numpy array representing a depth map
        :param intrinsic: A 3x3 numpy array representing camera intrinsic matrix
        :param camera_pose: A 4x4 transformation matrix representing the pose from world to camera
        """
        height, width = depth.shape
        if self.with_color and color:
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
        if self.with_color and color:
            surface_color = color[u, v]
        distance = surface_depth - volume_depth
        dist_filter = (
                (surface_depth > 0)
                * (distance >= -self.truncate_margin)
        ).bool()
        x, y, z = x[dist_filter], y[dist_filter], z[dist_filter]
        distance = distance[dist_filter]
        surface_depth = surface_depth[dist_filter]
        if self.with_color and color:
            surface_color = surface_color[dist_filter]
        dist_trunc = torch.clamp_max(distance/self.truncate_margin, 1)

        # update volume
        var_sensor = self.axial_noise_model(surface_depth) ** 2
        var_old = self.variance_volume[x, y, z]
        sdf_old = self.sdf_volume[x, y, z]
        p = var_sensor / (var_sensor + var_old)
        q = 1 - p
        self.sdf_volume[x, y, z] = p * sdf_old + q * dist_trunc
        self.variance_volume[x, y, z] = p * var_old
        if self.with_color and color:
            color_old = self.decode_color(self.color_volume[x, y, z])
            self.color_volume[x, y, z] = self.encode_color(p * self.decode_color(color_old) + q * self.decode_color(surface_color))


    def axial_noise_model(self, z):
        # """
        # The noise model is originated from 'Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking'.
        # :param z: The measured distances from camera to surface.
        # :return: The estimated standard deviations for each distance.
        # """
        # return 0.0012 + 0.0019 * (z - 0.4) ** 2

        """
        """
        return 0.00163 + 0.0007278 * z + 0.003949 * (z ** 2)
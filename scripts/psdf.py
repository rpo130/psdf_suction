import torch
import numpy as np


class PSDF:
    def __init__(self, shape, resolution, device="cuda:0", with_color=False):
        self.shape = np.array(shape)
        self.resolution = resolution
        self.truncate_margin = resolution * 5
        self.device = device
        self.with_color = with_color

        # initialization
        # (xvol, yvol, zvol)
        I, J, K = np.meshgrid(np.arange(self.shape[0]),
                                np.arange(self.shape[1]),
                                np.arange(self.shape[2]),
                                indexing='ij')
        # (xvol, yvol, zvol, 3)
        self.indices = torch.from_numpy(np.stack([I, J, K], axis=-1)).to(device)
        # (xvol, yvol, zvol, 3)
        self.positions = (self.indices.float() + 0.5) * self.resolution
        self.sdf = torch.ones(self.shape.tolist(), dtype=torch.float32, device=device)
        self.var = torch.ones(self.shape.tolist(), dtype=torch.float32, device=device) * 1e-3
        if with_color:
            self.rgb = torch.zeros(self.shape.tolist()+[3], dtype=torch.uint8, device=device)

    def camera2pixel(self, points, intrinsic):
        # points (xindex,yindex,zindex,3)
        # p (x/z, y/z, 1)
        # (xvol, yvol, zvol, 3) = (xvol, yvol, zvol, 3) * (3,3)
        uv_ = torch.round((points / points[..., 2:3]) @ intrinsic.T)
        # (xvol, yvol, zvol, 2)
        return uv_[..., :2].long()

    def fuse(self, depth, intrinsic, T_cam_to_vol, color=None, method='dynamic', beta=1):
        """

        :param depth: (h,w,)
        :param intrinsic: (3,3,)
        :param camera_pose:
        :param color:
        :param method: "dynamic"(default), "normal", "average"
        :param beta:
        :return:
        """

        height, width = depth.shape
        depth = torch.from_numpy(depth).to(self.device)
        if self.with_color and color is not None:
            color = torch.from_numpy(color).to(self.device)

        cam_intr = torch.FloatTensor(intrinsic).to(self.device)
        T_cam_to_vol = torch.FloatTensor(T_cam_to_vol).to(self.device)
        R_vol_to_cam = T_cam_to_vol[:3, :3].T
        t_vol_to_cam = - R_vol_to_cam @ T_cam_to_vol[:3, 3]

        # find all voxel within the camera view
        cam_coors = self.positions @ R_vol_to_cam.T + t_vol_to_cam
        img_coors = self.camera2pixel(cam_coors, cam_intr)
        valid_mask = (
                (0 <= img_coors[..., 0]) * (img_coors[..., 0] < width)
                * (0 <= img_coors[..., 1]) * (img_coors[..., 1] < height)
                * (cam_coors[..., 2] > 0)
        ).bool()
        voxel_coors = self.indices[valid_mask].long()
        pixel_coors = img_coors[valid_mask].long()
        x, y, z = voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2]
        #??? u, v
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
        sdf_new = torch.clamp_max(distance/self.truncate_margin, 1)
        var_new = self.axial_noise_model(surface_depth) ** 2
        if self.with_color and color is not None:
            rgb_new = surface_color[dist_filter]

        # update volume
        var_old = self.var[x, y, z]
        sdf_old = self.sdf[x, y, z]
        if method in ["dynamic", "normal"]:
            p = var_new / (var_new + var_old)
            q = 1 - p
            self.sdf[x, y, z] = p * sdf_old + q * sdf_new
            self.var[x, y, z] = p * var_old
            if method == "dynamic":
                sign = ((sdf_old * sdf_new) < 0).int()
                alpha, lambd = 1e-2, 0
                diff = torch.abs(sdf_old - sdf_new)
                var_c = sign * (torch.exp(alpha * diff) - 1) + (1 - sign) * lambd
                self.var[x, y, z] += var_c
            if self.with_color and color is not None:
                self.rgb[x, y, z] = (p[..., None] * self.rgb[x, y, z] + q[..., None] * rgb_new).type(torch.uint8)
        elif method == "average":
            self.sdf[x, y, z] = sdf_old + 1 / beta * (sdf_new - sdf_old)
            self.var[x, y, z] = 1e-5
            if self.with_color and color is not None:
                self.rgb[x, y, z] = (self.rgb[x, y, z] + 1 / beta * (rgb_new - self.rgb[x, y, z])).type(torch.uint8)
        else:
            assert(False and "wrong PSDF method")


    def axial_noise_model(self, z):
        return 0.00163 + 0.0007278 * z + 0.003949 * (z ** 2)


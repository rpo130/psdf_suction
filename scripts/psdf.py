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
        #https://dev.intelrealsense.com/docs/projection-in-intel-realsense-sdk-20#pixel-coordinates
        #camera coor is x point right, y point down, z point forward
        #depth row is height, col is width; height is in y-direction, col is in x-direction, so need change u,v location
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

    def fuse_point(self, point_in_world : np.ndarray, T_world_to_volume):
        """
        point shape=(3)
        T_world_to_volume shape=(4,4)
        """
        t_world_to_vol = T_world_to_volume[:3, 3]
        point_in_vol = point_in_world + t_world_to_vol

        xp,yp,zp = point_in_vol
        xmin = 0 * self.resolution
        xmax = self.shape[0] * self.resolution
        ymin = 0 * self.resolution
        ymax = self.shape[1] * self.resolution
        zmin = 0 * self.resolution
        zmax = self.shape[2] * self.resolution

        if xp < xmin or xp > xmax:
            print("x out")
            return
        if yp < ymin or yp > ymax:
            print("y out")
            return            
        if zp < zmin or zp > zmax:
            print("z out")
            return
            
        x = int(xp // self.resolution)
        y = int(yp // self.resolution)
        z = int(zp // self.resolution)

        sdf_new = 0
        var_new = 0
        var_old = self.var[x, y, z]
        sdf_old = self.sdf[x, y, z]
        p = var_new / (var_new + var_old)
        q = 1 - p
        self.sdf[x, y, z] = p * sdf_old + q * sdf_new
        self.var[x, y, z] = p * var_old

    def fuse_contact(self, T_contact_to_vol):
        T_contact_to_vol = torch.FloatTensor(T_contact_to_vol).to(self.device)
        R_vol_to_contact = T_contact_to_vol[:3, :3].T
        t_vol_to_contact = - R_vol_to_contact @ T_contact_to_vol[:3, 3]

        # find all voxel within the camera view
        contact_coors = self.positions @ R_vol_to_contact.T + t_vol_to_contact
        valid_mask = (
                (abs(contact_coors[..., 0]-0) < 0.001)
                * (abs(contact_coors[..., 1]-0) < 0.001)
        ).bool()
        voxel_coors = self.indices[valid_mask].long()
        x, y, z = voxel_coors[:, 0], voxel_coors[:, 1], voxel_coors[:, 2]

        # get truncated distance
        volume_depth = contact_coors[x, y, z, 2]
        surface_depth = 0
        distance = surface_depth - volume_depth
        dist_filter = (
                (distance >= -self.truncate_margin)
        ).bool()
        x, y, z = x[dist_filter], y[dist_filter], z[dist_filter]
        distance = distance[dist_filter]
        sdf_new = torch.clamp_max(distance/self.truncate_margin, 1)
        var_new = 0 * self.axial_noise_model(surface_depth) ** 2

        # update volume
        var_old = self.var[x, y, z]
        sdf_old = self.sdf[x, y, z]
        p = var_new / (var_new + var_old)
        q = 1 - p
        self.sdf[x, y, z] = p * sdf_old + q * sdf_new
        self.var[x, y, z] = p * var_old

    def get_point_cloud(self, T_volume_to_world=np.eye(4)):
        from skimage import measure

        verts, faces, _, _ = measure.marching_cubes_lewiner(self.sdf.cpu().numpy(), 0)
        return (verts * self.resolution) @ T_volume_to_world[:3, :3].T + T_volume_to_world[:3, 3], faces
    
    """
        return : 
            heightmap
            heightmap-normal
            heightmap-var
            heightmap-color
    """
    def flatten(self, smooth=False, ksize=5, sigmaColor=0.1, sigmaSpace=5):
        psdf = self
        # find surface point
        #(250,250,250)
        surface_mask = psdf.sdf <= 0.01
        # max in z direction
        surface_mask_flat = torch.max(surface_mask, dim=-1)[0]

        # get height map
        z_vol = torch.zeros_like(psdf.sdf).long()
        z_vol[surface_mask] = psdf.indices[surface_mask][:, 2]
        z_flat = torch.max(z_vol, dim=-1)[0]
        #(250,250)
        height_map = psdf.positions[..., 2].take(z_flat)
        if smooth:
            import cv2 as cv
            height_map = cv.bilateralFilter(height_map, ksize, sigmaColor, sigmaSpace)

        # get point map
        # (250,250,3)
        point_map = psdf.positions[:, :, 0, :].clone()
        point_map[..., 2] = height_map

        # get normal map
        normal_map = compute_surface_normal(point_map)

        # get variance map
        variances_map = psdf.var.take(z_flat)
        variances_map[~surface_mask_flat] = 10

        # get color map
        color_map = psdf.rgb.take(z_flat)

        # re-arrangement
        # variances_map = torch.flip(variances_map, dims=[0,1])
        # point_map = torch.flip(point_map, dims=[0,1])
        # # normals = torch.flip(normals, dims=[0,1])
        # color_map = torch.flip(color_map, dims=[0,1])

        return (point_map.cpu().numpy(), 
                normal_map.cpu().numpy(), 
                variances_map.cpu().numpy(), 
                color_map.cpu().numpy())

def compute_surface_normal(point_map):
    # (250, 250, 3)
    height, width, _ = point_map.shape
    s = 1

    # lower - upper
    coor_up = torch.zeros_like(point_map).to(point_map.device)
    coor_down = torch.zeros_like(point_map).to(point_map.device)
    coor_up[s:height, ...] = point_map[0:height - s, ...]
    coor_down[0:height - s, ...] = point_map[s:height, ...]
    # i(x+1) - i(x-1)
    dx = coor_down - coor_up

    # right - left
    coor_left = torch.zeros_like(point_map).to(point_map.device)
    coor_right = torch.zeros_like(point_map).to(point_map.device)
    coor_left[:, s:width, :] = point_map[:, 0:width - s, ...]
    coor_right[:, 0:width - s, :] = point_map[:, s:width, ...]
    dy = coor_right - coor_left

    # normal
    surface_normal = torch.cross(dx, dy, dim=-1)
    #(250, 250)
    norm_normal = torch.norm(surface_normal, dim=-1)
    norm_mask = (norm_normal == 0)
    surface_normal[norm_mask] = 0
    surface_normal[~norm_mask] = surface_normal[~norm_mask] / norm_normal[~norm_mask][:, None]
    return surface_normal
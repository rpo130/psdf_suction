import numpy as np
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import geometry_msgs.msg
import torch
from typing import Union

def get_orientation(origin_ori, normal):
    z_axis = np.array([0.0, 0.0, 1.0])
    rvec = np.cross(z_axis, normal)
    if np.linalg.norm(rvec) == 0:
        rvec = z_axis
    else:
        rvec = rvec / np.linalg.norm(rvec)
    theta = np.arccos(np.dot(z_axis, normal))
    mat = R.from_rotvec(rvec*theta).as_matrix()
    r = mat @ R.from_quat(origin_ori).as_matrix()
    return R.from_matrix(r).as_quat()

def transform2matrix(transform):
    translation = transform.translation
    rotation = transform.rotation
    T = np.eye(4)
    T[:3, :3] = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w]).as_dcm()
    T[:3, 3] = [translation.x, translation.y, translation.z]
    return T

def pose2mat(pose):
    if isinstance(pose, geometry_msgs.msg.Pose):
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = np.array([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])
    elif isinstance(pose, list):
        t = pose[:3]
        q = pose[3:]
    else:
        print("pose should in type ", geometry_msgs.msg.Pose, "or", list)
    T = np.eye(4)
    try:
        T[:3, :3] = R.from_quat(q).as_dcm()
    except Exception as e:
        T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T

def clip_joint(joint_positions):
    joint_positions = joint_positions % (np.pi * 2)
    filter = np.abs(joint_positions) > np.pi
    joint_positions[filter] = joint_positions[filter] - np.pi * 2
    return joint_positions

def numpy2multiarray(data):
    data_buffer = Float32MultiArray()
    for dim in data.shape:
        data_buffer.layout.dim.append(MultiArrayDimension(size=dim))
    data_buffer.data = data.reshape(-1).tolist()
    return data_buffer

def multiarray2numpy(data_buffer):
    shape = []
    for dim in data_buffer.layout.dim:
        shape.append(dim.size)
    return np.array(data_buffer.data, dtype=np.float32).reshape(shape)

def trilinear_interpolate_torch(idx, f):
    """
    h, w, d = f.shape
    0<=idx[0]<h
    0<=idx[1]<w
    0<=idx[2]<d
    """
    h, w, d = f.shape
    values = torch.zeros(idx.shape[0], dtype=f.dtype, device=f.device)
    i, j, k = idx[:, 0], idx[:, 1], idx[:, 2]
    u0, v0, w0 = torch.floor(i).long(), torch.floor(j).long(), torch.floor(k).long()
    # edge = torch.logical_or(torch.logical_or(u0 == h-1, v0 == w-1), w0 == d-1) # bottom and right edge
    edge = ((u0 == h-1) + (v0 == w-1) + (w0 == d-1)).bool()
    values[edge] = f[u0[edge], v0[edge], w0[edge]]

    u0, v0, w0 = u0[~edge], v0[~edge], w0[~edge]
    u, v, w = i[~edge]-u0, j[~edge]-v0, k[~edge]-w0
    values[~edge] = f[u0,v0,w0]*(1-u)*(1-v)*(1-w) + \
                    f[u0+1,v0,w0]*(u)*(1-v)*(1-w) + f[u0,v0+1,w0]*(1-u)*(v)*(1-w) + f[u0,v0,w0+1]*(1-u)*(1-v)*(w) + \
                    f[u0+1,v0+1,w0]*(u)*(v)*(1-w) + f[u0,v0+1,w0+1]*(1-u)*(v)*(w) + f[u0+1,v0,w0+1]*(u)*(1-v)*(w) + \
                    f[u0+1,v0+1,w0+1]*(u)*(v)*(w)
    return values


def get_orientation(origin_ori, normal):
    z_axis = np.array([0.0, 0.0, 1.0])
    rvec = np.cross(z_axis, normal)
    if np.linalg.norm(rvec) == 0:
        rvec = z_axis
    else:
        rvec = rvec / np.linalg.norm(rvec)
    theta = np.arccos(np.dot(z_axis, normal))
    mat = R.from_rotvec(rvec*theta).as_matrix()
    r = mat @ R.from_quat(origin_ori).as_matrix()
    return R.from_matrix(r).as_quat()

def compute_surface_normal(point_cloud):
        """
        outward image
        """
        height, width, _ = point_cloud.shape
        coor_up = np.zeros_like(point_cloud)
        coor_down = np.zeros_like(point_cloud)
        coor_left = np.zeros_like(point_cloud)
        coor_right = np.zeros_like(point_cloud)
        s = 1

        # lower - upper
        coor_up[s:height, ...] = point_cloud[0:height - s, ...]
        coor_down[0:height - s, ...] = point_cloud[s:height, ...]
        dx = coor_down - coor_up

        # right - left
        coor_left[:, s:width, :] = point_cloud[:, 0:width - s, ...]
        coor_right[:, 0:width - s, :] = point_cloud[:, s:width, ...]
        dy = coor_right - coor_left

        # normal
        surface_normal = np.cross(dx, dy, axis=-1)
        norm_normal = np.linalg.norm(surface_normal, axis=-1)
        norm_mask = norm_normal == 0
        surface_normal[norm_mask] = 0
        surface_normal[~norm_mask] = surface_normal[~norm_mask] / norm_normal[~norm_mask][:, None]
        return surface_normal

def get_point_cloud_from_depth(depth, intr):
    h, w = depth.shape[:2]
    id = np.stack(np.meshgrid(np.arange(h), np.arange(w), indexing='ij'), axis=-1)
    points_c = np.concatenate([id[..., ::-1], np.ones_like(id[..., [0]])], axis=-1)
    points_c = points_c @ np.linalg.inv(intr).T * depth.reshape(h, w, 1)
    return points_c, id

def compute_surface_normal_torch(point_cloud):
    """
    outward image
    """
    height, width, _ = point_cloud.shape
    coor_up = torch.zeros_like(point_cloud).to(point_cloud.device)
    coor_down = torch.zeros_like(point_cloud).to(point_cloud.device)
    coor_left = torch.zeros_like(point_cloud).to(point_cloud.device)
    coor_right = torch.zeros_like(point_cloud).to(point_cloud.device)
    s = 1

    # lower - upper
    coor_up[s:height, ...] = point_cloud[0:height - s, ...]
    coor_down[0:height - s, ...] = point_cloud[s:height, ...]
    dx = coor_down - coor_up
    # dx = coor_down - point_cloud
    # right - left
    coor_left[:, s:width, :] = point_cloud[:, 0:width - s, ...]
    coor_right[:, 0:width - s, :] = point_cloud[:, s:width, ...]
    dy = coor_right - coor_left
    # dy = coor_right - point_cloud

    # normal
    surface_normal = torch.cross(dx, dy, dim=-1)
    norm_normal = torch.norm(surface_normal, dim=-1)
    norm_mask = norm_normal == 0
    surface_normal[norm_mask] = 0
    surface_normal[~norm_mask] = surface_normal[~norm_mask] / norm_normal[~norm_mask][:, None]
    return surface_normal

def get_point_cloud_from_depth_torch(depth, intr):
    h, w = depth.shape[:2]
    id = torch.stack(torch.meshgrid(torch.arange(h).to(depth.device), torch.arange(w).to(depth.device)), dim=-1).float()
    points_c = torch.stack([id[..., 1], id[..., 0], torch.ones_like(id[..., 0])], dim=-1)
    points_c = points_c @ torch.inverse(intr).T * depth.reshape(h, w, 1)
    return points_c, id

"""
    vec: n...x3 matrix
    t: 4x4 matrix
"""
def apply_spatial_transformation(vec : Union[np.ndarray, torch.Tensor], t : Union[np.ndarray, torch.Tensor]):
    vec_new = vec @ t[:3,:3].T + t[:3,3]
    return vec_new

"""
:depth: HxW
:K: 相机内参
"""
def persp2ortho(depth2origin, K):
    H, W = depth2origin.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),np.arange(H, dtype=np.float32),
                       indexing='xy')
    #每一格偏移到中间
    #(x,y,1)
    dirs = np.stack([(i+0.5-K[0][2])/K[0][0],
                     -(j+0.5-K[1][2])/K[1][1],
                     -np.ones_like(i)],
                    -1)
    # depth2plane / 1 = depth2origin / norm((x,y,1))
    depth2plane = depth2origin / np.linalg.norm(dirs, axis=-1)
    return depth2plane


"""
:depth: HxW
:K: 相机内参
"""
def ortho2persp(depth2plane, K):
    H, W = depth2plane.shape
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),np.arange(H, dtype=np.float32),
                       indexing='xy')
    dirs = np.stack([(i+0.5-K[0][2])/K[0][0],
                     -(j+0.5-K[1][2])/K[1][1],
                     -np.ones_like(i)],
                    -1)
    # depth2plane / 1 = depth2origin / norm((x,y,1))
    depth2origin = depth2plane * np.linalg.norm(dirs, axis=-1)
    return depth2origin
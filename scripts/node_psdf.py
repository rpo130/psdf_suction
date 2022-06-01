#!/usr/bin/python3
import numpy as np
import json
import torch
from scipy.spatial.transform.rotation import Rotation as R
import cv2

import rospy
import rosparam
import sensor_msgs.msg
import geometry_msgs.msg
import message_filters

from psdf import PSDF
from configs import config

def compute_surface_normal(point_map):
    height, width, _ = point_map.shape
    s = 1

    # lower - upper
    coor_up = torch.zeros_like(point_map).to(point_map.device)
    coor_down = torch.zeros_like(point_map).to(point_map.device)
    coor_up[s:height, ...] = point_map[0:height - s, ...]
    coor_down[0:height - s, ...] = point_map[s:height, ...]
    dx = coor_down - coor_up

    # right - left
    coor_left = torch.zeros_like(point_map).to(point_map.device)
    coor_right = torch.zeros_like(point_map).to(point_map.device)
    coor_left[:, s:width, :] = point_map[:, 0:width - s, ...]
    coor_right[:, 0:width - s, :] = point_map[:, s:width, ...]
    dy = coor_right - coor_left

    # normal
    surface_normal = torch.cross(dx, dy, dim=-1)
    norm_normal = torch.norm(surface_normal, dim=-1)
    norm_mask = (norm_normal == 0)
    surface_normal[norm_mask] = 0
    surface_normal[~norm_mask] = surface_normal[~norm_mask] / norm_normal[~norm_mask][:, None]
    return surface_normal

def flatten(psdf, smooth=False, ksize=5, sigmaColor=0.1, sigmaSpace=5):

    # find surface point
    surface_mask = torch.abs(psdf.sdf) <= 0.01
    surface_mask_flat = torch.max(surface_mask, dim=-1)[0]

    # get height map
    z_vol = torch.zeros_like(psdf.sdf).long()
    z_vol[surface_mask] = psdf.indices[surface_mask][:, 2]
    z_flat = torch.max(z_vol, dim=-1)[0]
    height_map = psdf.positions[..., 2].take(z_flat)
    if smooth:
        height_map = cv2.bilateralFilter(height_map, ksize, sigmaColor, sigmaSpace)

    # get point map
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

    return point_map, normal_map, variances_map, color_map

def get_point_cloud(psdf):
    return psdf.positions[torch.abs(psdf.sdf) <= 0.01]

def main():
    rospy.init_node("psdf_suction/psdf")
    # show_point_cloud = rosparam.get_param("show_point_cloud")

    # initialize PSDF
    range = np.array([
        [config.x_min, config.x_max],
        [config.y_min, config.y_max],
        [config.z_min, config.z_max]
    ])
    psdf = PSDF(range, config.resolution, with_color=True)

    # load camera intrinsic and hand-eye-calibration
    with open("config/cam_info_realsense.json", 'r') as f:
        cam_info = json.load(f)
    cam_intr = np.array(cam_info["K"]).reshape(3, 3)
    cam_height = cam_info["height"]
    cam_width = cam_info["width"]
    T_cam_to_tool0 = np.array(cam_info["cam_to_tool0"]).reshape(4, 4)

    # publish "point map" which is required by analysis module
    point_map_pub = rospy.Publisher("psdf/point_map", sensor_msgs.msg.Image, queue_size=1)
    normal_map_pub = rospy.Publisher("psdf/normal_map", sensor_msgs.msg.Image, queue_size=1)
    variance_map_pub = rospy.Publisher("psdf/variance_map", sensor_msgs.msg.Image, queue_size=1)
    if show_point_cloud:
        point_cloud_pub = rospy.Publisher("psdf/point_cloud", sensor_msgs.msg.PointCloud, queue_size=1)

    # subscribe camera and robot pose
    # and fuse new data to PSDF
    def fuse_cb(depth_msg : sensor_msgs.msg.Image, 
                color_msg : sensor_msgs.msg.Image, 
                tool0_pose_msg : geometry_msgs.msg.Pose):
        assert(depth_msg.height == cam_height and depth_msg.width == cam_width)
        assert(color_msg.height == cam_height and color_msg.width == cam_width)
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).astype(np.float32).reshape(cam_height, cam_width) / 1000
        color = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(cam_height, cam_width, 3)
        
        p, q = tool0_pose_msg.position, tool0_pose_msg.orientation
        T_tool0_to_world = np.eye(4)
        T_tool0_to_world[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T_tool0_to_world[:3, 3] = [p.x, p.y, p.z]
        T_cam_to_world = T_tool0_to_world @ T_cam_to_tool0
        
        psdf.fuse(np.copy(depth), cam_intr, T_cam_to_world, color=np.copy(color))
        point_map, normal_map, variance_map, _ = flatten(psdf)

        point_map_pub.publish(
            sensor_msgs.msg.Image(
                data=point_map.cpu().numpy().tobytes(), 
                height=psdf.shape[0], 
                width=psdf.shape[1]
            )
        )
        normal_map_pub.publish(
            sensor_msgs.msg.Image(
                data=normal_map.cpu().numpy().tobytes(), 
                height=psdf.shape[0], 
                width=psdf.shape[1]
            )
        )
        variance_map_pub.publish(
            sensor_msgs.msg.Image(
                data=variance_map.cpu().numpy().tobytes(), 
                height=psdf.shape[0], 
                width=psdf.shape[1]
            )
        )
        if show_point_cloud:
            point_cloud = get_point_cloud(psdf).cpu().numpy()
            point_cloud_msg = sensor_msgs.msg.PointCloud()
            point_cloud_msg.header.frame_id = "base_link"
            point_cloud_msg.points = []
            for point in point_cloud:
                point_cloud_msg.points.append(geometry_msgs.msg.Point32(x=point[0], y=point[1], z=point[2]))
            point_cloud_pub.publish(point_cloud_msg)

    depth_sub = message_filters.Subscriber(cam_info["depth_topic"], sensor_msgs.msg.Image)
    color_sub = message_filters.Subscriber(cam_info["color_topic"], sensor_msgs.msg.Image)
    tool0_sub = message_filters.Subscriber(cam_info["tool0_pose_topic"], geometry_msgs.msg.Pose)
    sub_syn = message_filters.ApproximateTimeSynchronizer([depth_sub, color_sub, tool0_sub], 1e10, 1e-3, allow_headerless=True)
    sub_syn.registerCallback(fuse_cb)

    print("PSDF running")
    rospy.spin()


if __name__=="__main__":
    main()
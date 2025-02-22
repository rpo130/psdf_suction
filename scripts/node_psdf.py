#!/usr/bin/python3
import os
import numpy as np
import json
from scipy.spatial.transform.rotation import Rotation as R

from psdf import PSDF
from configs import config, DEVICE
import configs

def main():
    import rospy
    import rosparam
    import std_msgs.msg
    import sensor_msgs.msg
    import geometry_msgs.msg
    import message_filters

    rospy.init_node("psdf")
    rosnamespace = configs.getnamespace()
    show = rosparam.get_param(rospy.get_name() + "/show")
    method = rosparam.get_param(rospy.get_name() + "/method")

    # initialize PSDF
    psdf = PSDF(config.volume_shape, config.volume_resolution, device=DEVICE, with_color=True)

    # load camera intrinsic and hand-eye-calibration
    with open(os.path.join(os.path.dirname(__file__), "../config/cam_info_realsense.json"), 'r') as f:
        cam_info = json.load(f)
    cam_intr = np.array(cam_info["K"]).reshape(3, 3)
    cam_height = cam_info["height"]
    cam_width = cam_info["width"]

    # publish "point map" which is required by analysis module
    point_map_pub = rospy.Publisher(rospy.get_name() + "/point_map", sensor_msgs.msg.Image, queue_size=1)
    normal_map_pub = rospy.Publisher(rospy.get_name() + "/normal_map", sensor_msgs.msg.Image, queue_size=1)
    variance_map_pub = rospy.Publisher(rospy.get_name() + "/variance_map", sensor_msgs.msg.Image, queue_size=1)
    if show:
        point_cloud_pub = rospy.Publisher(rospy.get_name() + "/point_cloud", sensor_msgs.msg.PointCloud2, queue_size=1)
        point_image_pub = rospy.Publisher(rospy.get_name() + "/point_image", sensor_msgs.msg.Image, queue_size=1)
        normal_image_pub = rospy.Publisher(rospy.get_name() + "/normal_image", sensor_msgs.msg.Image, queue_size=1)
        variance_image_pub = rospy.Publisher(rospy.get_name() + "/variance_image", sensor_msgs.msg.Image, queue_size=1)
        depth_cloud_pub = rospy.Publisher(rospy.get_name() + "/depth_cloud", sensor_msgs.msg.PointCloud2, queue_size=1)
        rospy.loginfo("PSDF show topic:")
        rospy.loginfo("\t" + point_cloud_pub.resolved_name)
        rospy.loginfo("\t" + point_image_pub.resolved_name)
        rospy.loginfo("\t" + normal_image_pub.resolved_name)
        rospy.loginfo("\t" + variance_image_pub.resolved_name)
        rospy.loginfo("\t" + depth_cloud_pub.resolved_name)

    # subscribe camera and robot pose
    # and fuse new data to PSDF
    def fuse_cb(depth_msg : sensor_msgs.msg.Image, 
                color_msg : sensor_msgs.msg.Image, 
                camera_pose_msg : geometry_msgs.msg.PoseStamped):
        assert(depth_msg.height == cam_height and depth_msg.width == cam_width)
        assert(color_msg.height == cam_height and color_msg.width == cam_width)
        #(height,width)
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).astype(np.float32).reshape(cam_height, cam_width) / 1000
        color = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(cam_height, cam_width, 3)
        
        p, q = camera_pose_msg.pose.position, camera_pose_msg.pose.orientation
        T_cam_to_world = np.eye(4)
        T_cam_to_world[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        T_cam_to_world[:3, 3] = [p.x, p.y, p.z]
        T_cam_to_volume = config.T_world_to_volume @ T_cam_to_world
        
        # fuse new data
        ts = rospy.rostime.get_time()
        psdf.fuse(np.copy(depth), cam_intr, T_cam_to_volume, color=np.copy(color), method=method)
        rospy.loginfo("fuse time %f"%(rospy.rostime.get_time()-ts))

        # flatten to 2D point image
        ts = rospy.rostime.get_time()
        point_map, normal_map, variance_map, _ = psdf.flatten()
        rospy.loginfo("flatten time %f"%(rospy.rostime.get_time()-ts))

        # publishing topic message
        ts = rospy.rostime.get_time()
        point_map_pub.publish(sensor_msgs.msg.Image(
            data=point_map.tobytes(), height=psdf.shape[0], width=psdf.shape[1]
        ))
        normal_map_pub.publish(sensor_msgs.msg.Image(
            data=normal_map.tobytes(), height=psdf.shape[0], width=psdf.shape[1]
        ))
        variance_map_pub.publish(sensor_msgs.msg.Image(
            data=variance_map.tobytes(), height=psdf.shape[0], width=psdf.shape[1]
        ))
        rospy.loginfo("publish time %f"%(rospy.rostime.get_time()-ts))
        if show:
            ts = rospy.rostime.get_time()
            # point cloud
            point_cloud = psdf.get_point_cloud(config.T_volume_to_world).astype(np.float32)
            point_size = point_cloud.dtype.itemsize * 3
            point_cloud_msg = sensor_msgs.msg.PointCloud2(
                header=std_msgs.msg.Header(frame_id="base_link", stamp=rospy.Time.now()),
                fields=[sensor_msgs.msg.PointField(name=name,
                                                    offset=point_cloud.dtype.itemsize*i, 
                                                    datatype=sensor_msgs.msg.PointField.FLOAT32, 
                                                    count=1) for i, name in enumerate("xyz")],
                data=point_cloud.tobytes(),
                height=point_cloud.shape[0],
                width=1,
                point_step=point_size,
                row_step=point_size,
                is_dense=False,
                is_bigendian=False
            )
            point_cloud_pub.publish(point_cloud_msg)
            # point image
            point_image_pub.publish(sensor_msgs.msg.Image(
                data=((point_map[..., 2] - config.T_volume_to_world[2, 3]) / config.volume_range[2]).tobytes(),
                height=psdf.shape[0], width=psdf.shape[1],
                encoding="32FC1"
            ))
            # normal image
            normal_image_pub.publish(sensor_msgs.msg.Image(
                data=((normal_map[..., 2] + 1) / 2).tobytes(),
                height=psdf.shape[0], width=psdf.shape[1],
                encoding="32FC1"
            ))
            # variance image
            variance_image_pub.publish(sensor_msgs.msg.Image(
                data=((variance_map - variance_map.min()) / (variance_map.max() - variance_map.min())).tobytes(),
                height=psdf.shape[0], width=psdf.shape[1],
                encoding="32FC1"
            ))
            # depth cloud
            I, J = np.meshgrid(np.arange(depth.shape[0]), np.arange(depth.shape[1]), indexing="ij")
            depth_cloud = np.stack([J, I, np.ones_like(I)], axis=-1) @ np.linalg.inv(cam_intr).T * depth[..., None]
            depth_cloud = depth_cloud.reshape(-1, 3)
            depth_cloud = (depth_cloud @ T_cam_to_world[:3, :3].T + T_cam_to_world[:3, 3]).astype(np.float32)
            point_size = depth_cloud.dtype.itemsize * 3
            depth_cloud_msg = sensor_msgs.msg.PointCloud2(
                header=std_msgs.msg.Header(frame_id="base_link", stamp=rospy.Time.now()),
                fields=[sensor_msgs.msg.PointField(name=name,
                                                    offset=depth_cloud.dtype.itemsize*i, 
                                                    datatype=sensor_msgs.msg.PointField.FLOAT32, 
                                                    count=1) for i, name in enumerate("xyz")],
                data=depth_cloud.tobytes(),
                height=depth_cloud.shape[0],
                width=1,
                point_step=point_size,
                row_step=point_size,
                is_dense=False,
                is_bigendian=False
            )
            depth_cloud_pub.publish(depth_cloud_msg)
            rospy.loginfo("show time %f"%(rospy.rostime.get_time()-ts))

    queue_size = 1
    depth_sub = message_filters.Subscriber(cam_info["depth_topic"], 
        sensor_msgs.msg.Image, queue_size=queue_size, buff_size=queue_size*cam_height*cam_width*2)
    color_sub = message_filters.Subscriber(cam_info["color_topic"], 
        sensor_msgs.msg.Image, queue_size=queue_size, buff_size=queue_size*cam_height*cam_width*3)
    tool0_sub = message_filters.Subscriber(rosnamespace + "camera_pose", 
        geometry_msgs.msg.PoseStamped, queue_size=queue_size)
    sub_syn = message_filters.ApproximateTimeSynchronizer(
        [depth_sub, color_sub, tool0_sub], 10, 1)
    sub_syn.registerCallback(fuse_cb)

    rospy.loginfo("PSDF running")
    rospy.spin()


if __name__=="__main__":
    main()
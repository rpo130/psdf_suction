#!/usr/bin/python2
import os
from turtle import position, stamp  
import numpy as np
import json
from scipy.spatial.transform.rotation import Rotation as R

import rospy
import tf2_ros
import tf.transformations as tf_trans
import geometry_msgs.msg
import std_msgs.msg

from configs import config

if __name__ == '__main__':
    rospy.init_node('cam_pose')
    rate = rospy.Rate(125)

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)

    cam_pose_pub = rospy.Publisher(rospy.get_namespace() + 'camera_pose', geometry_msgs.msg.PoseStamped, queue_size=1)
    with open(os.path.join(config.path, "config/cam_info_realsense.json"), 'r') as f:
        cam_info = json.load(f)
        T_cam_to_tool0 = np.array(cam_info["cam_to_tool0"]).reshape(4, 4)

    while not rospy.is_shutdown():
        try:
            trans = tfBuffer.lookup_transform('base_link', 'tool0', rospy.Time())
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue
        
        q = trans.transform.rotation
        q = np.array([q.x, q.y, q.z, q.w])
        t = trans.transform.translation
        t = np.array([t.x, t.y, t.z])
        T_tool0_to_world = np.eye(4)
        T_tool0_to_world[:3, :3] = R.from_quat(q).as_dcm()
        T_tool0_to_world[:3, 3] = t

        T_cam_to_world = np.matmul(T_tool0_to_world, T_cam_to_tool0)
        q = R.from_dcm(T_cam_to_world[:3, :3]).as_quat()
        t = T_cam_to_world[:3, 3]
        cam_pose_pub.publish(geometry_msgs.msg.PoseStamped(
            header=std_msgs.msg.Header(frame_id="base_link", stamp=trans.header.stamp),
            pose=geometry_msgs.msg.Pose(
                position=geometry_msgs.msg.Point(x=t[0], y=t[1], z=t[2]),
                orientation=geometry_msgs.msg.Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            )
        ))

        rate.sleep()
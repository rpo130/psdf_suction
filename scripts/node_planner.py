#!/usr/bin/python3
import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
import cv2

import rospy
import rosparam
import sensor_msgs.msg
import message_filters
import psdf_suction.srv

from analyser.vacuum_cup_analyser import VacuumCupAnalyser
from configs import config, DEVICE, EPSILON

def compute_score(graspable_map, point_map, normal_map, position_pre,
                    dist_weight_pub=None, normal_weight_pub=None, range_weight_pub=None):
    # current first
    dist_sigma = 0.05
    dist_weight = np.exp(-0.5 * (((point_map - position_pre) ** 2).sum(axis=-1) / dist_sigma ** 2))
    dist_weight = dist_weight / (dist_weight.sum() + EPSILON)

    # upward first
    normal_weight = normal_map[..., 2] + 1
    normal_weight = normal_weight / (normal_weight.sum() + EPSILON)

    # face center first
    ksize = 21
    range_weight = graspable_map * cv2.GaussianBlur(graspable_map, (ksize, ksize), 5)
    range_weight = range_weight / (range_weight.sum() + EPSILON)
    
    # show topic
    if dist_weight_pub is not None:
        dist_weight_pub.publish(sensor_msgs.msg.Image(
            data=dist_weight.tobytes(), height=dist_weight.shape[0], width=dist_weight.shape[1]
        ))
    if normal_weight_pub is not None:
        normal_weight_pub.publish(sensor_msgs.msg.Image(
            data=normal_weight.tobytes(), height=normal_weight.shape[0], width=normal_weight.shape[1]
        ))
    if range_weight_pub is not None:
        range_weight_pub.publish(sensor_msgs.msg.Image(
            data=range_weight.tobytes(), height=range_weight.shape[0], width=range_weight.shape[1]
        ))

    score = dist_weight * normal_weight * range_weight
    return score

def main():
    rospy.init_node("planner")
    show_grasp_point = rosparam.get_param(rospy.get_name() + "/show")


    # show 
    if show_grasp_point:
        graspable_map_pub = rospy.Publisher(rospy.get_name() + "/graspable_map", sensor_msgs.msg.Image, queue_size=1)
        score_pub = rospy.Publisher(rospy.get_name() + "/score", sensor_msgs.msg.Image, queue_size=1)
        dist_weight_pub = rospy.Publisher(rospy.get_name() + "/dist_weight", sensor_msgs.msg.Image, queue_size=1)
        normal_weight_pub = rospy.Publisher(rospy.get_name() + "/normal_weight", sensor_msgs.msg.Image, queue_size=1)
        range_weight_pub = rospy.Publisher(rospy.get_name() + "/range_weight", sensor_msgs.msg.Image, queue_size=1)
        rospy.loginfo("Planner show topic:")
        rospy.loginfo("\t" + graspable_map_pub.resolved_name)
        rospy.loginfo("\t" + score_pub.resolved_name)
        rospy.loginfo("\t" + dist_weight_pub.resolved_name)
        rospy.loginfo("\t" + normal_weight_pub.resolved_name)
        rospy.loginfo("\t" + range_weight_pub.resolved_name)
    
    # planning log
    global planning
    global position_pre
    global grasp_position
    global grasp_normal
    global request_mask
    planning = False
    position_pre = None
        
    # analyse and choose grasp pose
    def planning_cb(point_map_msg, normal_map_msg, variance_map_msg):
        global planning
        global position_pre
        global grasp_position
        global grasp_normal
        global request_mask

        if planning is False or position_pre is None:
            rospy.rostime.wallsleep(0.01)
            return
        planning = False

        height, width = point_map_msg.height, point_map_msg.width
        point_map = np.frombuffer(point_map_msg.data, dtype=np.float32).reshape(height, width, 3).copy()
        normal_map = np.frombuffer(normal_map_msg.data, dtype=np.float32).reshape(height, width, 3).copy()
        variance_map = np.frombuffer(variance_map_msg.data, dtype=np.float32).reshape(height, width).copy()

        # find point need to be analysed
        normal_mask = normal_map[..., 2] > np.cos(config.gripper_angle_threshold/180*np.pi)
        variance_mask = variance_map < 1e-2
        final_mask = normal_mask * variance_mask * request_mask

        # model based grasp pose analysis
        analyser = VacuumCupAnalyser(radius=config.gripper_radius, 
                                    height=config.gripper_height, 
                                    num_vertices=config.gripper_vertices,
                                    angle_threshold=config.gripper_angle_threshold)
        vision_dict = {"point_cloud": point_map,
                       "normal": normal_map}
        obj_ids = np.where(final_mask != 0)
        graspable_map = analyser.analyse(vision_dict, obj_ids)

        # choosing from grasp pose candidates
        position_pre = np.zeros(3)
        if show_grasp_point:
            score = compute_score(graspable_map, point_map, normal_map, position_pre,
                                    dist_weight_pub, normal_weight_pub, range_weight_pub)
            height = config.volume_shape[0]
            width = config.volume_shape[1]
            graspable_map_pub.publish(sensor_msgs.msg.Image(data=graspable_map.tobytes(), height=height, wdith=width))
            score_pub.publish(sensor_msgs.msg.Image(data=score.tobytes(), height=height, wdith=width))
        else:
            score = compute_score(graspable_map, point_map, normal_map, position_pre)
        idx = np.argmax(score)
        i, j = idx // width, idx % width
        grasp_position = point_map[i, j].tolist()
        grasp_normal = normal_map[i, j].tolist()
        
    image_size = config.volume_shape[0] * config.volume_shape[1]
    point_map_sub = message_filters.Subscriber(rospy.get_namespace() + "psdf/point_map", 
        sensor_msgs.msg.Image, queue_size=1, buff_size=2*image_size*4*3)
    normal_map_sub = message_filters.Subscriber(rospy.get_namespace() + "psdf/normal_map", 
        sensor_msgs.msg.Image, queue_size=1, buff_size=2*image_size*4*3)
    variance_map_sub = message_filters.Subscriber(rospy.get_namespace() + "psdf/variance_map", 
        sensor_msgs.msg.Image, queue_size=1, buff_size=2*image_size*4)
    sub_syn = message_filters.ApproximateTimeSynchronizer(
        [point_map_sub, normal_map_sub, variance_map_sub], 1e10, 1e-3, allow_headerless=True)
    sub_syn.registerCallback(planning_cb)

    # service for getting new grasp pose
    def get_grasp_pose_cb(req : psdf_suction.srv.GetGraspPoseRequest):
        global planning
        global position_pre
        global request_mask
        global grasp_position
        global grasp_normal
        print("get_grasp_pose request")
        position_pre = np.array(req.previous_grasp_position)
        request_mask = np.frombuffer(req.mask.data, np.uint8).reshape(req.mask.height, req.mask.width)
        planning = True
        while planning:
            rospy.rostime.wallsleep(0.01)
        res = psdf_suction.srv.GetGraspPoseResponse()
        res.grasp_position = grasp_position
        res.grasp_normal = grasp_normal
        print("get_grasp_pose response")
        return res

    rospy.Service(rospy.get_name() + "/get_grasp_pose", psdf_suction.srv.GetGraspPose, get_grasp_pose_cb)

    rospy.loginfo("Planner running")
    rospy.spin()

if __name__=="__main__":
    main()
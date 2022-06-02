#!/usr/bin/python3
import numpy as np
import torch
from scipy.spatial.transform.rotation import Rotation as R
import cv2

import rospy
import rosparam
import sensor_msgs.msg
import geometry_msgs.msg
import message_filters
import psdf_suction.srv

from analyser.vacuum_cup_analyser import VacuumCupAnalyser
from background import BackgroundExtractor
from configs import config, DEVICE, EPSILON

def compute_score(graspable_map, point_map, normal_map, position_pre):
    # weighting candidate grasps
    graspable_map = torch.FloatTensor(graspable_map).to(DEVICE)
    point_map = torch.FloatTensor(point_map).to(DEVICE)
    normal_map = torch.FloatTensor(point_map).to(DEVICE)
    position_pre = torch.FloatTensor(position_pre).to(DEVICE)

    # current first
    dist_sigma = 0.05
    dist_weight = torch.FloatTensor([1 / (2 * np.pi * dist_sigma ** 2) ** 1.5]).to(DEVICE) \
                    * torch.exp(-0.5 * (((point_map - position_pre) ** 2).sum(dim=-1) / dist_sigma ** 2))
    dist_weight = dist_weight / (dist_weight.sum() + EPSILON)

    # upward first
    normal_weight = normal_map[..., 2] + 1
    normal_weight = normal_weight / (normal_weight.sum() + EPSILON)

    # face center first
    ksize = 21
    gau_k = cv2.getGaussianKernel(ksize, 5)
    gau_f = torch.FloatTensor(gau_k * gau_k.T).to(DEVICE)
    area_weight = torch.conv2d(graspable_map[None, None, ...], gau_f[None, None, ...], stride=1, padding=ksize//2)
    area_weight *= graspable_map
    area_weight = area_weight[0, 0, ...] / (area_weight.sum() + EPSILON)
    
    return dist_weight * normal_weight * area_weight

def main():
    rospy.init_node("planner")
    show_grasp_point = rosparam.get_param("/psdf_suction/planner/show_grasp_point")

    # vacuum cup model analyser
    analyser = VacuumCupAnalyser(radius=config.gripper_radius, 
                                 height=config.gripper_height, 
                                 num_vertices=config.gripper_vertices,
                                 angle_threshold=config.gripper_angle_threshold)

    # background
    background_extractor = BackgroundExtractor()

    # exploring strategy
    global planning
    global position_pre
    global grasp_position
    global grasp_normal
    global request_mask
    planning = False
    position_pre = None
    target_grasp_pose = geometry_msgs.msg.Pose()

    def planning_cb(point_map_msg, normal_map_msg, variance_map_msg):
        global planning
        global position_pre
        global grasp_position
        global grasp_normal
        global request_mask
        if planning is False or position_pre is None:
            rospy.sleep(0.001)
            return

        height, width = point_map_msg.height, point_map_msg.width
        point_map = np.frombuffer(point_map_msg.data, dtype=np.float32).reshape(height, width, 3)
        normal_map = np.frombuffer(normal_map_msg.data, dtype=np.float32).reshape(height, width, 3)
        variance_map = np.frombuffer(variance_map_msg.data, dtype=np.float32).reshape(height, width)

        # find point need to be analysed
        obj_mask = background_extractor.extract(point_map)
        normal_mask = normal_map[..., 2] > torch.cos(torch.tensor(config.gripper_angle_threshold/180*np.pi)).to(DEVICE)
        variance_mask = variance_map < 1e-2
        final_mask = normal_mask * variance_mask * request_mask

        # model based grasp pose analysis
        vision_dict = {"point_cloud": point_map,
                       "normal": normal_map}
        obj_ids = np.where(final_mask != 0)
        graspable_map, _ = analyser.analyse(vision_dict, obj_ids)

        # choosing from grasp pose candidates
        position_pre = np.zeros(3)
        score = compute_score(graspable_map, point_map, normal_map, position_pre)
        idx = np.argmax(score)
        i, j = idx // width, idx % width
        grasp_position = point_map[i, j]
        grasp_normal = normal_map[i, j]
        
        planning = False

    point_map_sub = message_filters.Subscriber("/psdf_suction/point_map", sensor_msgs.msg.Image)
    normal_map_sub = message_filters.Subscriber("/psdf_suction/normal_map", sensor_msgs.msg.Image)
    variance_map_sub = message_filters.Subscriber("/psdf_suction/variance_map", sensor_msgs.msg.Pose)
    sub_syn = message_filters.ApproximateTimeSynchronizer(
        [point_map_sub, normal_map_sub, variance_map_sub], 1e10, 1e-3, allow_headerless=True)
    sub_syn.registerCallback(planning_cb)

    # service for getting new grasp pose
    def get_grasp_pose_cb(req):
        global planning
        global position_pre
        global request_mask
        global grasp_position
        global grasp_normal
        position_pre = np.array(req.previous_grasp_position)
        request_mask = np.frombuffer(req.mask.data, np.uint8).reshape(req.mask.height, req.mask.width)
        planning = True
        while planning:
            rospy.sleep(0.001)
        res = psdf_suction.srv.GetGraspPoseResponse()
        res.grasp_position = grasp_position
        res.grasp_normal = grasp_normal
        return res

    rospy.Service("/psdf_suction/get_grasp_pose", psdf_suction.srv.GetGraspPose, get_grasp_pose_cb)

    rospy.logdebug("Planner running")
    rospy.spin()

if __name__=="__main__":
    main()
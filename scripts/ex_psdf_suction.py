#!/usr/bin/python3
import os
import sys
from urllib import response
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from scipy.spatial.transform import Rotation as R

sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import rospy
import rosparam
import psdf_suction.srv

from configs import config
from vaccum_cup import VaccumCup
from ur5_commander import UR5Commander
from recorder import Recorder
from analyser.vacuum_cup_analyser import VacuumGripper
from utils import get_orientation

method_name = "main"

target_name = "level_1"
# target_name = "level_2"

# trials_num = "test"
trials_num = "test"

experiment_dir = os.path.join("experiment", method_name, target_name, trials_num)

DEVICE = 'cuda:0'
EPSILON = 1e-6

if __name__ == '__main__':
    rospy.init_node('psdf_suction')
    rate = rospy.Rate(10)
    cfg = rosparam.get_param

    # suction cup init
    print("[ init suction cup ]")
    cup = VaccumCup()

    # arm control init
    print("[ init arm control ]")
    arm = UR5Commander()
    static_positions, _ = rosparam.load_file("config/positions.yaml")[0]
    # print(static_positions)

    # camera info
    with open("config/cam_info_realsense.json", 'r') as f:
        cam_info = json.load(f)

    # PSDF-Suction planner
    func_get_grasp_pose = rospy.ServiceProxy("/psdf_suction/get_grasp_pose", psdf_suction.srv.GetGraspPose)

    # recorder
    print("[ init recorder ]")
    recorder = Recorder(experiment_dir)
    recorder.start()
    arm.set_positions(static_positions['place_pre_positions'], wait=True)
    recorder.set_target_region()
    cnt_success = 0
    cnt_failed = 0
    cnt_failed_cum = 0

    # grasp pose init
    init_grasp_position = np.array([0.5 * (config.x_max + config.x_min), 
                                    0.5 * (config.y_max + config.x_min),
                                    config.z_min])
    init_grasp_normal = np.array([0., 0., 1.])
    init_distance = config.init_cam_height - init_grasp_position[2]
    
    # failure map
    failure_map = np.zeros((config.psdf_length, config.psdf_width), dtype=np.float)
    idx = np.stack(np.meshgrid(np.arange(config.psdf_length), np.arange(config.psdf_width), indexing='ij'), axis=-1)
    failure_region = 5
    failure_decay = 1.0

    # start grasp loop
    while True:

        # go to init positions
        arm.set_positions(static_positions['init_positions'], wait=True)
        rospy.sleep(1)

        # set start time
        recorder.set()

        # control loop
        grasp_normal = np.copy(init_grasp_position)
        grasp_position = np.copy(init_grasp_normal)
        distance = init_distance
        z_axis = torch.FloatTensor([[0, 0, 1]]).to(DEVICE)
        while True:
            seg_mask = np.ones((150,150))

            req = psdf_suction.srv.GetGraspPoseRequest()
            req.previous_grasp_position = grasp_position.tolist()
            req.mask = seg_mask.tobytes()
            res = func_get_grasp_pose(req)

            grasp_position = np.array(res.grasp_position)
            grasp_normal = np.array(res.grasp_normal)
            grasp_orientation = get_orientation(config.init_quaternion, grasp_normal)

            if distance <= max(0.1 + config.vacuum_length, 0.2):
                # execute grasp
                grasp_pre_pose = (grasp_position + grasp_normal * (0.1 + config.vacuum_length)).tolist() + grasp_orientation.tolist()
                arm.set_pose(grasp_pre_pose, wait=True)
                grasp_pose = (grasp_position + grasp_normal * config.vacuum_length).tolist() + grasp_orientation.tolist()
                arm.set_pose(grasp_pose, wait=True)
                break
            else:
                # transform to tool0 pose
                T_cam2world = np.eye(4)
                T_cam2world[:3, :3] = R.from_quat(grasp_orientation).as_matrix()
                T_cam2world[:3, 3] = grasp_position + grasp_normal * distance
                T_tool02world = T_cam2world @ np.linalg.inv(cam_info["cam_to_tool0"])
                move_position = T_tool02world[:3, 3]
                move_orientation = R.from_matrix(T_tool02world[:3, :3]).as_quat()
                move_pose = move_position.tolist() + move_orientation.tolist()
                arm.set_pose(move_pose, wait=False)
                distance -= 0.02


        # place process
        cup.grasp()
        grasp_postpose = arm.get_pose()
        grasp_postpose[2] = 0.4
        arm.set_pose(grasp_postpose, wait=True)
        arm.set_positions(static_positions["place_pre_positions"], wait=True)
        success, metric = recorder.check()
        arm.set_positions(static_positions["place_positions"], wait=True)
        cup.release()

        # logging
        grasp_num = cnt_failed+cnt_success
        np.savez(os.path.join(experiment_dir, "point_map_{}".format(grasp_num)), points_w.cpu().numpy())
        np.savetxt(os.path.join(experiment_dir, "grasp_{}.txt".format(grasp_num)), np.array(grasp_position.tolist() + grasp_orientation.tolist()))
        if success:
            cnt_success += 1
            cnt_failed_cum = 0
            recorder.log("[ grasp {} ]".format(grasp_num), success, metric, "success:", cnt_success, "failed", cnt_failed)
            if cnt_success == 15:
                recorder.log("[ finished ]", 15, "success, suppose all object is grasped")
                break
        else:
            cnt_failed += 1
            cnt_failed_cum += 1
            recorder.log("[ grasp {} ]".format(grasp_num), success, metric, "success:", cnt_success, "failed", cnt_failed)
            if cnt_failed_cum == 10:
                recorder.log("[ finished ]", 10, "continue failure, stop grasping")
                break
            new_map = 1/(2*np.pi*failure_region**2)*np.exp(-0.5*((id[..., 0]-i)**2+(id[..., 1]-j)**2)/(2*failure_region**2))
            new_map = (new_map-new_map.min()) / (new_map.max() - new_map.min())
            failure_map[new_map > failure_map] = new_map[new_map > failure_map]
        if (cnt_failed + cnt_success) == 45:
            recorder.log("[ finished ]", 45, "grasp is executed, stop grasping")
            break
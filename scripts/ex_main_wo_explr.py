#!/usr/bin/env python
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import rospy
import rosparam

from scripts.config import config
from scripts.utils import get_orientation
from scripts.vaccum_cup import VaccumCup
from scripts.ur5_commander import UR5Commander
from scripts.psdf_commander import PSDFCommander
from scripts.recorder import Recorder
from scripts.analysis.vacuum_gripper import VacuumGripper

method_name = "main_wo_explr"

# target_name = "level_1"
target_name = "level_2"

# trials_num = "test"
trials_num = "9"

experiment_dir = os.path.join("../experiment", method_name, target_name, trials_num)

DEVICE = 'cuda:0'
EPSILON = 1e-6

if __name__ == '__main__':
    rospy.init_node('policy_local_normal')
    rate = rospy.Rate(10)
    cfg = rosparam.get_param

    # suction cup init
    print("[ init suction cup ]")
    cup = VaccumCup()

    # arm control init
    print("[ init arm control ]")
    arm = UR5Commander()
    static_positions, _ = rosparam.load_file("../config/positions.yaml")[0]
    # print(static_positions)

    # psdf perception init
    print("[ init perception module ]")
    T_cam2tool0 = np.loadtxt("../config/camera_pose_base_tip.txt")
    bounds = np.stack([config.lower_bound, config.upper_bound]).T
    psdf = PSDFCommander(arm, bounds, config.resolution, DEVICE)

    # recorder
    print("[ init recorder ]")
    recorder = Recorder(experiment_dir)
    recorder.start()
    arm.set_positions(static_positions['place_pre_positions'], wait=True)
    recorder.set_target_region()

    # segmentation tool
    print("[ setting segmentation background ]")
    seg_threshold_depth = 0.015
    seg_threshold_color = 0.1
    arm.set_positions(static_positions['init_positions'], wait=True)
    print("set backgound and press key")
    input()
    for i in range(10):
        theta = i/10*2*np.pi
        normal = np.array([np.cos(theta), np.sin(theta), 2])
        normal = normal / np.linalg.norm(normal)
        ori = get_orientation(config.init_quaternion, normal)
        pose = config.init_position.tolist() + ori.tolist()
        arm.set_pose(pose, wait=True)
    arm.set_positions(static_positions['init_positions'], wait=True)
    back_points, _, _, back_colors = psdf.get_data()
    print("set object and press key")
    input()
    plt.figure("background")
    plt.imshow(back_points[..., 2].cpu().numpy())
    # print(back_colors.min(), back_colors.max())
    # plt.figure("background_color")
    # plt.imshow(back_colors.cpu().numpy()/255)
    # plt.waitforbuttonpress()

    # analysis tool
    gripper = VacuumGripper(config.gripper_radius,
                            config.gripper_height,
                            config.gripper_vertices,
                            config.gripper_angle_threshold / 180 * np.pi)

    # start grasp loop
    cnt_success = 0
    cnt_failed = 0
    cnt_failed_cum = 0
    plt.ion()
    failure_map = np.zeros(back_points.shape[:2], dtype=np.float)
    id = np.stack(np.meshgrid(np.arange(back_points.shape[0]), np.arange(back_points.shape[1]), indexing='ij'), axis=-1)
    failure_region = 5
    failure_decay = 1.0
    while True:

        # go to init positions
        arm.set_positions(static_positions['init_positions'], wait=True)
        rospy.sleep(1)

        # set start time
        recorder.set()
        z_axis = torch.FloatTensor([[0, 0, 1]]).to(DEVICE)

        # get scene data
        points_w, normals_w, variances, colors = psdf.get_data()
        seg = points_w[..., 2] > (back_points[..., 2] + seg_threshold_depth)
        seg = seg * (torch.norm((colors - back_colors)/255, dim=-1)/3 > 0)
        seg = seg * (points_w[..., 2] != 0)
        seg = seg * (back_points[..., 2] != 0)
        plt.figure("seg")
        plt.imshow(seg.cpu().numpy())
        # plt.waitforbuttonpress()

        # get threshold map
        normal_mask = (normals_w @ z_axis.T) > torch.cos(torch.tensor(config.gripper_angle_threshold/180*np.pi)).to(DEVICE)
        normal_mask = normal_mask * (normals_w[..., [2]] > 0)
        variance_mask = variances < 1e-2
        workspace_mask = (points_w[..., 0] >= config.x_min) * (points_w[..., 0] <= config.x_max) \
                       * (points_w[..., 1] >= config.y_min) * (points_w[..., 1] <= config.y_max) \
                       * (points_w[..., 2] >= config.z_min) * (points_w[..., 2] <= config.z_max)
        valid = seg * normal_mask[..., -1] * workspace_mask * variance_mask * torch.FloatTensor(failure_map < 0.5).to(DEVICE)
        if valid.sum() == 0:
            valid = seg * normal_mask[..., -1] * workspace_mask * variance_mask
        valid = valid.bool()
        # plt.figure("valid")
        # plt.imshow(valid.cpu().numpy())
        plt.figure("failure_map")
        plt.imshow(failure_map)
        # plt.waitforbuttonpress()

        if valid.sum() != 0:
            # analyze candidate points
            vision_dict = {}
            vision_dict.update({'point_cloud': points_w.cpu().numpy()})
            vision_dict.update({'normal': -normals_w.cpu().numpy()})
            score, _ = gripper.update_visiondict(vision_dict, np.where(valid.cpu().numpy()))
        else:
            print("[ no valid point ]")
            score = np.zeros(valid.shape, dtype=float)
        # plt.figure("score")
        # plt.imshow(score)
        # plt.waitforbuttonpress()

        # weighting candidate grasps
        score = torch.FloatTensor(score).to(DEVICE)
        act_map = torch.ones_like(score)

        # upward first
        normal_weight = torch.abs(torch.arccos(normals_w @ z_axis.T))[..., -1]
        normal_weight = normal_weight.max() - normal_weight
        normal_weight = normal_weight / (normal_weight.sum() + EPSILON)
        act_map *= normal_weight

        # face center first
        ksize = 21
        gau_k = cv2.getGaussianKernel(ksize, 5)
        gau_f = torch.FloatTensor(gau_k * gau_k.T).to(DEVICE)
        if score.sum() != 0:
            area_weight = score * \
                          torch.conv2d(score[None, None, ...], gau_f[None, None, ...], stride=1, padding=ksize//2)
        else:
            print("[ no graspable point ]")
            area_weight = seg.float() * \
                          torch.conv2d(seg.float()[None, None, ...], gau_f[None, None, ...], stride=1, padding=ksize//2)
        area_weight = area_weight[0, 0, ...] / (area_weight.sum() + EPSILON)
        # print(area_weight.sum())
        act_map *= area_weight
        plt.figure("area weight")
        plt.imshow(area_weight.cpu().numpy())
        # plt.figure("seg")
        # plt.imshow(seg.float().cpu().numpy())
        # plt.waitforbuttonpress()

        # score = metric_local_normal_variance(points_w, normals_w, seg)
        # if (score * (1-failure_map)).sum() != 0:
        #     score = score * (1-failure_map)
        scores = (act_map * valid).cpu().numpy()
        idx = np.argmax(scores)
        i, j = idx//points_w.shape[1], idx%points_w.shape[1]
        grasp_position = points_w[i, j].cpu().numpy()
        grasp_normal = normals_w[i, j].cpu().numpy()
        grasp_orientation = get_orientation(config.init_quaternion, grasp_normal)
        print(grasp_position)
        print(grasp_normal)
        print(i, j)
        plt.ion()
        plt.show()
        # plt.waitforbuttonpress(1)

        # execute grasp
        grasp_pre_pose = (grasp_position + grasp_normal * (0.1 + config.vacuum_length)).tolist() + grasp_orientation.tolist()
        arm.set_pose(grasp_pre_pose, wait=True)
        grasp_pose = (grasp_position + grasp_normal * config.vacuum_length).tolist() + grasp_orientation.tolist()
        arm.set_pose(grasp_pose, wait=True)

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

        # print(i, j)
        # print(grasp_normal)
        # plt.figure("color")
        # plt.imshow(color)
        # plt.figure("depth")
        # plt.imshow(depth)
        # plt.figure("seg")
        # plt.imshow(seg)
        # plt.figure("normal")
        # plt.imshow(normals_w)
        # plt.figure("score")
        # plt.imshow(score)
        # plt.figure("valid")
        # plt.imshow(valid)
        # plt.figure("fail")
        # plt.imshow(failure_map)
        # plt.waitforbuttonpress()

    psdf.close()
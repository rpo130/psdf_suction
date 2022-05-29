#!/usr/bin/env python
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')

import cv2
import os
from perception import BinaryImage, CameraIntrinsics, ColorImage, DepthImage
import matplotlib.pyplot as plt
import numpy as np
import torch


import rospy
import rosparam

from scripts.config import config
from scripts.utils import *
from scripts.vaccum_cup import VaccumCup
from scripts.ur5_commander import UR5Commander
from scripts.realsense_commander import RealSenseCommander
from scripts.recorder import Recorder
from gqcnn.srv import GQCNNGraspPlannerSegmask

# method_name = "dex_net_4"
method_name = "dex_net_4_finetune"

target_name = "level_1"
# target_name = "level_2"

trials_num = "9"

experiment_dir = os.path.join("../experiment", method_name, target_name, trials_num)

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

    # realsense camera init
    print("[ init camera ]")
    cam = RealSenseCommander()
    cam.start()
    T_cam2tool0 = np.loadtxt("../config/camera_pose_base_tip.txt")
    camera_intr = CameraIntrinsics.load("../config/realsense.intr")
    cam_intr = np.array(camera_intr.K)

    # recorder
    print("[ init recorder ]")
    recorder = Recorder(experiment_dir)
    recorder.start()
    arm.set_positions(static_positions['place_pre_positions'], wait=True)
    recorder.set_target_region()

    # segmentation tool
    print("[ setting segmentation background ]")
    seg_threshold_color = 0.1
    seg_threshold_depth = 0.005
    arm.set_positions(static_positions['init_positions'], wait=True)
    print("set backgound and press key")
    input()
    back_color, back_depth = cam.get_image()
    back_depth = DepthImage(back_depth).inpaint(0.5).data
    print("set object and press key")
    input()

    # gqcnn related
    gqcnn_service_name = "gqcnn/grasp_planner_segmask"
    rospy.wait_for_service(gqcnn_service_name)
    plan_grasp_segmask = rospy.ServiceProxy(gqcnn_service_name, GQCNNGraspPlannerSegmask)
    print(gqcnn_service_name, "ready")

    # start grasp loop
    cnt_success = 0
    cnt_failed = 0
    cnt_failed_cum = 0
    plt.ion()
    failure_map = np.zeros((480, 640), dtype=np.float)
    failure_region = 10
    failure_decay = 1.0
    while True:

        # go to init positions
        arm.set_positions(static_positions['init_positions'], wait=True)

        # set start time
        recorder.set()

        # get image and segmentation
        color, depth = cam.get_image()
        depth = DepthImage(depth).inpaint(0.5).data
        diff_color = color.astype(np.float) - back_color.astype(np.float)
        diff_depth = depth.astype(np.float) - back_depth.astype(np.float)
        seg = np.logical_and(np.linalg.norm(diff_color/255, axis=-1)/3 > 0,
                             np.abs(diff_depth) > seg_threshold_depth)
        seg = np.logical_and(seg,
                             depth != 0)
        seg = np.logical_and(seg,
                             back_depth != 0)

        # get grasp action from policy
        T_tool02world = pose2mat(arm.get_pose())
        T_cam2world = T_tool02world @ T_cam2tool0
        points_c, id = get_point_cloud_from_depth(depth, cam_intr)
        points_w = points_c @ T_cam2world[:3, :3].T + T_cam2world[:3, 3]
        normals_w = compute_surface_normal(points_w)
        normal_mask = normals_w @ np.array([[0, 0, 1]]).T > np.cos(45/180*np.pi)
        workspace_mask = (points_w[..., 0] >= config.x_min) * (points_w[..., 0] <= config.x_max) \
                       * (points_w[..., 1] >= config.y_min) * (points_w[..., 1] <= config.y_max) \
                       * (points_w[..., 2] >= config.z_min) * (points_w[..., 2] <= config.z_max)
        valid = normal_mask[..., -1] * seg * workspace_mask * (failure_map < 0.5)
        if valid.sum() == 0:
            print("all fail")
            valid = normal_mask[..., -1] * seg * workspace_mask
        if valid.sum() == 0:
            print("no valid point")

        # gqcnn request
        depth_im = DepthImage(depth, frame=camera_intr.frame).inpaint(0.5)
        color_im = ColorImage(color, frame=camera_intr.frame)
        segmask = BinaryImage((valid*255).astype(np.uint8), frame=camera_intr.frame)
        grasp_resp = plan_grasp_segmask(color_im.rosmsg, depth_im.rosmsg, camera_intr.rosmsg, segmask.rosmsg)
        # print(grasp_resp)
        pose = grasp_resp.grasp.pose
        # print(pose)
        grasp_position_cam = np.array([pose.position.x, pose.position.y, pose.position.z])
        grasp_normal_cam = - R.from_quat(
            [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        ).as_matrix()[:, 0]
        grasp_position = grasp_position_cam @ T_cam2world[:3, :3].T + T_cam2world[:3, 3]
        grasp_normal = grasp_normal_cam @ T_cam2world[:3, :3].T
        grasp_orientation = get_orientation(config.init_quaternion, grasp_normal)
        j, i, _ = grasp_position_cam/grasp_position_cam[2] @ cam_intr.T
        # grasp_position = points_w[int(i), int(j)]
        # grasp_normal = normals_w[int(i), int(j)]
        # grasp_orientation = get_orientation(config.init_quaternion, grasp_normal)

        print(i, j)
        print(grasp_position)
        print(grasp_normal)
        print(grasp_resp.grasp.angle)
        # plt.figure("color")
        # plt.imshow(color)
        # plt.figure("depth")
        # plt.imshow(depth)
        # plt.figure("seg")
        # plt.imshow(seg)
        # plt.waitforbuttonpress()

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

        grasp_num = cnt_failed+cnt_success
        cv2.imwrite(os.path.join(experiment_dir, "color_{}.png".format(grasp_num)), color[..., ::-1])
        cv2.imwrite(os.path.join(experiment_dir, "seg_{}.png".format(grasp_num)), (seg * workspace_mask * 255).astype(np.uint8))
        np.savez(os.path.join(experiment_dir, "depth_{}".format(grasp_num)), depth)
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
        # plt.figure("normal")
        # plt.imshow(normals_w)
        # plt.figure("score")
        # plt.imshow(score)
        # plt.figure("valid")
        # plt.imshow(valid)
        # plt.figure("fail")
        # plt.imshow(failure_map > 0.5)
        # plt.waitforbuttonpress()
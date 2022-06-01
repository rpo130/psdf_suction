from curses import flash
import numpy as np
import torch

import rospy
import sensor_msgs.msg
import geometry_msgs.msg
import message_filters
import psdf_suction.srv

from analyser.vacuum_cup_analyser import VacuumCupAnalyser
from segmenter import Segmenter
from configs import config

def compute_score():
    # weighting candidate grasps
    score = torch.FloatTensor(score).to(DEVICE)
    act_map = torch.ones_like(score)

    # current first
    dist_sigma = 0.05
    dist_weight = torch.FloatTensor([1 / (2 * np.pi * dist_sigma ** 2) ** 1.5]).to(DEVICE) \
                    * torch.exp(-0.5 * (((points_w - torch.from_numpy(grasp_position).to(DEVICE)) ** 2).sum(dim=-1) / dist_sigma ** 2))
    dist_weight = dist_weight / (dist_weight.sum() + EPSILON)
    act_map *= dist_weight
    plt.figure("dist weight")
    plt.imshow(dist_weight.cpu().numpy())
    plt.waitforbuttonpress()

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

def main():
    rospy.init_node("planner")

    # background segmentation
    segmenter = Segmenter()

    # vacuum cup model analyser
    analyser = VacuumCupAnalyser(radius=config.gripper_radius, 
                                 height=config.gripper_height, 
                                 num_vertices=config.gripper_vertices,
                                 angle_threshold=config.gripper_angle_threshold)

    # exploring strategy
    global planning
    global position_pre
    global target_grasp_pose
    planning = False
    position_pre = None
    target_grasp_pose = geometry_msgs.msg.Pose()
    def planning_cb(point_map_msg, normal_map_msg, variance_map_msg):
        global planning
        global position_pre
        if planning is False or position_pre is None:
            rospy.sleep(0.001)
            return

        height, width = point_map_msg.height, point_map_msg.width
        point_map = np.frombuffer(point_map_msg.data, dtype=np.float32).reshape(height, width, 3)
        normal_map = np.frombuffer(normal_map_msg.data, dtype=np.float32).reshape(height, width, 3)
        variance_map = np.frombuffer(variance_map_msg.data, dtype=np.float32).reshape(height, width)

        # model based grasp pose analysis
        vision_dict = {"point_cloud": point_map,
                       "normal": normal_map}
        obj_mask = segmenter.segment(point_map)
        mask = (variance_map < 1e-2) * (obj_mask != 0)
        obj_ids = np.where(mask != 0)
        graspable_map, _ = analyser.analyse(vision_dict, obj_ids)

        # choosing from grasp pose candidates
        position_pre = np.zeros(3)
        score_map = compute_score(graspable_map, point_map, normal_map, position_pre)
        idx = np.argmax(score_map)
        i, j = idx // width, idx % width
        grasp_position = point_map[i, j]
        grasp_normal = normal_map[i, j]

        # build grasp pose
        grasp_orientation = get_orientation(config.init_quaternion, grasp_normal)
        target_grasp_pose.position.x = grasp_position[0]
        target_grasp_pose.position.y = grasp_position[1]
        target_grasp_pose.position.z = grasp_position[2]
        target_grasp_pose.orientation.x = grasp_orientation[0]
        target_grasp_pose.orientation.y = grasp_orientation[1]
        target_grasp_pose.orientation.z = grasp_orientation[2]
        target_grasp_pose.orientation.w = grasp_orientation[3]
        
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
        position_pre = np.array(req.previous_grasp_position)
        planning = True
        while planning:
            rospy.sleep(0.001)
        res = psdf_suction.srv.GetGraspPoseResponse()
        res.target_grasp_pose = target_grasp_pose
        return res
    rospy.Service("/psdf_suction/get_grasp_pose", psdf_suction.srv.GetGraspPose, get_grasp_pose_cb)

    rospy.spin()

if __name__=="__main__":
    main()
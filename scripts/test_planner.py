#!/usr/bin/python3
import numpy as np
from pandas import wide_to_long

import rospy
import sensor_msgs.msg
import psdf_suction.srv

from configs import config

def main():
    rospy.init_node("test_planner")

    rospy.wait_for_service("/psdf_suction/planner/get_grasp_pose")
    func_get_grasp_pose = rospy.ServiceProxy("/psdf_suction/planner/get_grasp_pose", psdf_suction.srv.GetGraspPose)

    while not rospy.is_shutdown():
        input()
        req = psdf_suction.srv.GetGraspPoseRequest()
        req.previous_grasp_position = np.array([0, 0, 0])
        req.mask = sensor_msgs.msg.Image(
            data=np.ones((config.volume_shape[0], config.volume_shape[1]), dtype=np.uint8).tobytes(),
            height=config.volume_shape[0], width=config.volume_shape[1]
        )
        res = func_get_grasp_pose(req)
        print("get new grasp pose")
        print(res.grasp_position)
        print(res.grasp_normal)

if __name__=="__main__":
    main()

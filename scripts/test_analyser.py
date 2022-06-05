import numpy as np
import torch

import rospy
import std_msgs.msg

from analyser.vacuum_cup_analyser import VacuumCupAnalyser, cuda


if __name__=="__main__":
    rospy.init_node("test_analyser")

    analyser = VacuumCupAnalyser(0.01, 0.02)
    
    def test_cb(msg):
        points = np.ones((250, 250, 3))
        normals = np.zeros_like(points)
        mask = np.zeros_like(points[..., 0])
        mask[100:200, 100:200] = 1
        obj_ids = np.where(mask > 0)
        graspable_map = analyser.analyse({"point_cloud": points, "normal": normals}, obj_ids)
        print("hello?")

    rospy.Subscriber("/test_analyser_topic", std_msgs.msg.Bool, test_cb)
    rospy.spin()
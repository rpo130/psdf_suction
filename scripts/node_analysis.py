import numpy as np

import rospy
import sensor_msgs.msg

import psdf_suction.srv
from scripts.analyser.vacuum_cup_analyser import VacuumCupAnalyser
from configs import config



def main():
    rospy.init_node("analysis")

    analyser = VacuumCupAnalyser(radius=config.gripper_radius, 
                                 height=config.gripper_height, 
                                 num_vertices=config.gripper_vertices,
                                 angle_threshold=config.gripper_angle_threshold)

    def get_graspable_map_cb(req):
        height, width = req.mask.height, req.mask.width
        point_map = np.frombuffer(req.point_map.data, dtype=np.float32).reshape(height, width, 3)
        normal_map = np.frombuffer(req.normal_map.data, dtype=np.float32).reshape(height, width, 3)
        mask = np.frombuffer(req.mask.data, dtype=np.uint8).reshape(height, width)
        
        vision_dict = { "point_cloud": point_map,
                        "normal": normal_map}
        obj_ids = np.where(mask != 0)
        graspable_map, _ = analyser.analyse(vision_dict, obj_ids)

        res = psdf_suction.srv.GetGraspableMapResponse()
        res.graspable_map = sensor_msgs.msg.Image(data=graspable_map.tobytes(), 
                                                  height=height, 
                                                  width=width)
        return res
    rospy.Service("/psdf_suction/get_graspable_map", psdf_suction.srv.GetGraspableMap, get_graspable_map_cb)

    rospy.spin()

if __name__=="__main__":
    main()
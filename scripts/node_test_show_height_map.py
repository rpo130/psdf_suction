import cv2
import numpy as np

import rospy
import sensor_msgs.msg

def main():
    rospy.init_node("test_show_height_map_node")

    def show_height_map(point_map_msg):
        point_map = np.frombuffer(point_map_msg.data, dtype=np.float32).reshape(point_map_msg.height, point_map_msg.width, 3)
        cv2.imshow("height_map", point_map[..., 2])
        cv2.waitKey(1)
    rospy.Subscriber("/psdf/point_map", sensor_msgs.msg.Image, show_height_map)

    rospy.spin()

if __name__=="__main__":
    main()

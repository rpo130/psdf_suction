#!/usr/bin/python3
import rospy
import geometry_msgs.msg
import std_msgs.msg

def main():
    rospy.init_node("test_pose_node")
    rate = rospy.Rate(125)

    tool0_pose_pub = rospy.Publisher("tool0_pose", geometry_msgs.msg.PoseStamped)

    while not rospy.is_shutdown():
        tool0_pose = geometry_msgs.msg.PoseStamped()
        tool0_pose.pose.orientation.w = 1
        tool0_pose.header = std_msgs.msg.Header(frame_id="base_link", stamp=rospy.Time.now())
        tool0_pose_pub.publish(tool0_pose)
        rate.sleep()

if __name__=="__main__":
    main()

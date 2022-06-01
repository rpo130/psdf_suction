import rospy
import geometry_msgs.msg

def main():
    rospy.init_node("test_pose_node")

    tool0_pose_pub = rospy.Publisher("tool0_pose", geometry_msgs.msg.Pose)

    while True:
        tool0_pose = geometry_msgs.msg.Pose()
        tool0_pose.orientation.w = 1
        tool0_pose_pub.publish(tool0_pose)

if __name__=="__main__":
    main()

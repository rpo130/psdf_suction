
# Dependencies
## [ROS melodic](http://wiki.ros.org/melodic)
  - [moveit-melodic](http://docs.ros.org/en/melodic/api/moveit_tutorials/html/index.html)
  - [Universal_Robots_ROS_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)
  - [realsense-ros](https://github.com/IntelRealSense/realsense-ros#installation-instructions)

## python3 packages
* rospy
* message_filters
* numpy
* pytorch
* pytorch3d (install by pip in linux only support CUDA 10.1, recommended [install from github](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md#building--installing-from-source))
* pycuda
* scikit-image

# Installation
- download code
```bash
git clone https://github.com/tungkw/psdf_suction.git
```

# Quick start

## run ros environment
```bash
source /path/to/catkin_ws/devel/setup.bash
# UR5
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:='255.255.255.255'
# moveit
roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch
# moveit python3 adapter
roslaunch moveit_py3 moveit_py3.launch
# Realsense
roslaunch realsense2_camera rs_aligned_depth.launch
```

## run PSDF-Suction
```bash
cd /path/to/psdf_suction
python3 scripts/ex_main.py
```

# Debug

- if find no controller, set 'name' in 'ur5_moveit_config/config/controllers.yaml' as "scaled_pos_joint_traj_controller", which is recommended [here](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/issues/55#issuecomment-562215033)
- set 'move_group/trajectory_execution/allowed_start_tolerance'
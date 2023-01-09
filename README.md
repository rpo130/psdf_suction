
# Dependencies
## CUDA 10.2
## ROS [melodic](http://wiki.ros.org/melodic)
  - [moveit1-melodic](http://docs.ros.org/en/melodic/api/moveit_tutorials/html/index.html)
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

## 1.build dependencies
## 1.1 ROS
- ROS-melodic
- [moveit1-melodic](http://docs.ros.org/en/melodic/api/moveit_tutorials/html/doc/getting_started/getting_started.html#install-moveit)
- [Universal Robot](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)
- realsense
  - [SDK v2.50](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) with librealsense2-dev
  - download ROS package which is compatible with version of SDK, ubuntu and ROS
## 1.2 Python
- python2
  ```bash
  python2 -m pip install scipy
  ```
- python3
  ```bash
  python3 -m pip install torch==1.2.0 # compatible with cuda https://pytorch.org/get-started/previous-versions/
  python3 -m pip install scikit-image==0.15.0 pycuda scipy
  ```

## 2. build PSDF-Suction 
```bash
# make a new ros workspace
mkdir -p ～/psdf_suction_ws/src
cd ~/psdf_suction_ws/src

# moveit_py3
git clone https://github.com/tungkw/moveit_py3.git

# PSDF Suction
git clone https://github.com/tungkw/psdf_suction.git

# build
cd ..
source ~/psdf_suction_dependency/devel/setup.bash
catkin config --cmake-args -DPYTHON_EXECUTABLE=`which python3`
catkin build
```

# Quick start

## 1. run ros dependencies
```bash
source ~/psdf_suction_ws/devel/setup.bash
# UR5
roslaunch ur_robot_driver ur5_bringup.launch robot_ip:='255.255.255.255'
# moveit
roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch
# moveit python3 adapter
roslaunch moveit_py3 moveit_py3.launch
# Realsense
roslaunch realsense2_camera rs_aligned_depth.launch
```

## 2. show PSDF
```bash
# run psdf node
roslaunch psdf_suction psdf.launch show:=True
# show in rviz
roscd psdf_suction
rosrun rviz rviz -d config/psdf_point_cloud.rviz
```

## 3. run PSDF-Suction
```bash
cd /path/to/psdf_suction
python3 scripts/ex_main.py
```

# Debug

- if find no controller, set 'name' in 'ur5_moveit_config/config/controllers.yaml' as "scaled_pos_joint_traj_controller", which is recommended [here](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/issues/55#issuecomment-562215033)
- set 'move_group/trajectory_execution/allowed_start_tolerance'
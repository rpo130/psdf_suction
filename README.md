# Updating
see README_.md

# Documentation
## Required Environment
* [ROS](http://wiki.ros.org/Documentation)

Here is the [tutorial](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment) that tells us how to create a ros workspace

## Required ROS packages
* [Universal_Robots_ROS_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)
* [universal robot](https://github.com/fmauch/universal_robot)
* [RealSense ROS Wrapper](https://github.com/IntelRealSense/realsense-ros#installation-instructions)

## Required python packages (in python3)
* numpy
* torch
* pytorch3d (install by pip in linux only support CUDA 10.1, recommended [install from github](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md#building--installing-from-source))
* pycuda
* scikit-image
* rospy
* message_filters

## Setup our own package
1.Clone the code to the src folder
    
    $ cd src
      git clone https://github.com/caijunhao/psdf.git
      cd ..
      catkin_make

## Quick start

### build

install this package using python3 environment

    $ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

### run

Please refer to relevant packages to see the details of installation. 
After everything getting done, here are the steps to control the ur5 and generate mesh

1.Start UR driver

    $ roslaunch ur_robot_driver ur5_bringup.launch \ 
      robot_ip:=192.168.1.2 kinematics_config:="${HOME}/my_robot_calibration.yaml"
 
   load the urcap program from the teach pendant, and press start button.

2.Launch moveit planning api

    $ roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch

(if find no controller, set 'name' in 'ur5_moveit_config/config/controllers.yaml' as "scaled_pos_joint_traj_controller", which is recommended [here](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/issues/55#issuecomment-562215033))

(set 'move_group/trajectory_execution/allowed_start_tolerance')

3.Launch Rviz visualization tool

    $ roslaunch ur5_moveit_config moveit_rviz.launch config:=true
    
4.Start RealSense Node
    
    $ roslaunch realsense2_camera rs_aligned_depth.launch

5.Launch main.launch to start controlling the robot

    $ roslaunch psdf main.launch
Node: The launch file contains two nodes, one is for controlling the robot, 
another one is used to evaluate the transformation 
from base link to camera link and publish it.

6.Launch psdf.launch to start psdf module
    
    $ roslaunch psdf psdf.launch

After launching all these nodes, you should turn back to the Rviz window, 
click the next button to move the robot, 
and enable the psdf module to start processing the data.

At the end, the psdf node will be shutdown and a mesh will be generated. 
You can use [Meshlab](https://www.meshlab.net/) to get the visualization of mesh.
 
 
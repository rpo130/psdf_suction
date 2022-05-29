# Updating
- 2020/10/16
    1. "scripts/main_node.py"
        - get new data from psdf_node
        - analyse and predict the grasp pose
        - send target pose to move_group_node
    2. dependency
        - cv_bridge
- 2020/10/14
    1. "scripts/move_group_commander.py"
        - movement handling
    2. "scripts/psdf_commander.py"
        - requesting height map and controlling start/stop flag
    3. quick start 2.Launch moveit planning api
        - need to set 'move_group/trajectory_execution/allowed_start_tolerance'
    4. "analysis/vacuum_gripper_gpu.py"
- 2020/10/13
    1. "scripts/move_group_helper.py"
        - move_group python api 
        - should run in python2
    2. "quick start 2.Launch moveit planning api" problem fixed
    3. "scripts/sdf_gpu.py"
        - "PSDF::get_volume()" add return "variance_volume"
    4. "scripts/psdf.py" -> "scripts/psdf_node.py"
        - add service "get_point_image" return filtered point cloud image
    5. add "scripts/predict.py" predict target pose and exploit move_group to make the movement
    6. requirement install comment

# Documentation
## Required Environment
* [ROS](http://wiki.ros.org/Documentation)

Here is the [tutorial](http://wiki.ros.org/ROS/Tutorials/InstallingandConfiguringROSEnvironment) that tells us how to create a ros workspace

## Required ROS packages
* [Universal_Robots_ROS_Driver](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver)
    - Universal_Robots_Clients_Library
    - industrial_robot_status_controller
    - industrial_core
    - ros_controllers
    
// * [universal robot](https://github.com/fmauch/universal_robot)

* [RealSense ROS Wrapper](https://github.com/IntelRealSense/realsense-ros#installation-instructions)
    - ddynamic_reconfigure
        - dynamic_reconfigure
    - realsense sdk
        - version matching
* cv_bridge (if use in python3, remove the ros build-in python2 library path before import, example can be found in "scripts/psdf_node.py")

## Required python packages (in python3)
* numpy
* torch
* pytorch3d (install by pip in linux only support CUDA 10.1, recommended [install from github](https://github.com/facebookresearch/pytorch3d/blob/master/INSTALL.md#building--installing-from-source))
* pycuda
* scikit-image
* rospy
* message_filters


## Quick start

### build

install this package using python3 environment

    $ catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3

### run

Please refer to relevant packages to see the details of installation (including the calibration step).
 
After everything getting done, here are the steps to control the ur5 and generate mesh

1. Start UR driver

        $ roslaunch ur_robot_driver ur5_bringup.launch \ 
          robot_ip:=192.168.1.2 kinematics_config:="${HOME}/my_robot_calibration.yaml"
     
       load the urcap program from the teach pendant, and press start button.

2. Launch moveit planning api

        $ roslaunch ur5_moveit_config ur5_moveit_planning_execution.launch
    
    (if find no controller, set 'name' in 'ur5_moveit_config/config/controllers.yaml' as "scaled_pos_joint_traj_controller", which is recommended [here](https://github.com/UniversalRobots/Universal_Robots_ROS_Driver/issues/55#issuecomment-562215033))
    
    (set 'move_group/trajectory_execution/allowed_start_tolerance')

3. Start RealSense Node
    
        $ roslaunch realsense2_camera rs_aligned_depth.launch

4. Launch psdf_node.launch to start psdf module
    
        $ roslaunch psdf psdf_node.launch
    
    Node: The launch file contains two nodes, one is for integrating sequential image, 
    another one is used to evaluate the transformation 
    from base link to camera link and publish it.
    
5. Launch main.launch to start controlling the robot

        $ roslaunch psdf main_node.launch

    will chase the target forever......
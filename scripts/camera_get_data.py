from mimetypes import init
from ur5_commander import UR5Commander
import rospy
from configs import config
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import json
from realsense_commander import RealSenseCommander
from matplotlib import pyplot as plt

import os
import numpy as np
import json
import matplotlib.pyplot as plt

T_scene_to_world = np.eye(4)
T_scene_to_world[:3, 3] = [0.46,0.05,0.03]

def pose_to_transform_matrix(pose, T_cam_to_tool0):
    tool0_pose = pose
    T_tool0_to_world = np.eye(4)
    T_tool0_to_world[:3, :3] = R.from_quat(tool0_pose[3:]).as_matrix()
    T_tool0_to_world[:3, 3] = tool0_pose[:3]
    T_cam_to_world = T_tool0_to_world @ T_cam_to_tool0
    return T_cam_to_world

def transform_matrix_to_pose(T_cam_to_world, T_cam_to_tool0):
    T_tool0_to_cam = np.linalg.inv(T_cam_to_tool0)
    T_tool0_to_world = T_cam_to_world @ T_tool0_to_cam
    tool0_pose = [] + T_tool0_to_world[:3,3].tolist() + R.from_matrix(T_tool0_to_world[:3,:3]).as_quat().tolist()
    return tool0_pose

def transform_matrix_to_pose(T_):
    pose = [] + T_[:3,3].tolist() + R.from_matrix(T_[:3,:3]).as_quat().tolist()
    return pose

def display():
    camera = RealSenseCommander()
    camera.start()
    print("Camera initialized")
    fig = plt.figure()

    while True:
        # get camera message
        color, depth = camera.get_image()
        plt.imshow(color)
        plt.show()

def move_and_get_image(cam_info, arm, camera):
    cam_intr = np.array(cam_info["K"]).reshape(3, 3)
    T_cam_to_tool0 = np.array(cam_info["cam_to_tool0"]).reshape(4, 4)

    #cam pose 
    obs_pose = gen_scene_obs_pose(T_scene_to_world)

    imgs = []
    depths = []
    poses = []
    for i, p in enumerate(obs_pose):
        arm.set_pose(transform_matrix_to_pose(p), wait=True)
        print(f'{i}')
        color, depth = camera.get_image()
        # print(f'color:{color}')
        # print(f'depth:{depth}')
        imgs.append(color)
        depths.append(depth)
        poses.append(pose_to_transform_matrix(arm.get_pose(), T_cam_to_tool0))
    gen_data(imgs, depths, poses, cam_intr)

def gen_data(imgs,depths,poses,cam_intr):
    basedir = "avt_data"
    imagedir = 'images'
    npdir = 'np'

    tranforms = {}
    tranforms['fx'] = cam_intr[0,0]
    tranforms['fy'] = cam_intr[1,1]
    tranforms['cx'] = cam_intr[0,2]
    tranforms['cy'] = cam_intr[1,2]
    #fov in degree
    #angle in rad
    #https://www.intel.com/content/www/us/en/support/articles/000030385/emerging-technologies/intel-realsense-technology.html
    #d435i hfov = 69
    tranforms['camera_angle_x'] = 69 / 180 * np.pi 
    frames = []
    tranforms['frames'] = frames

    if not os.path.exists(os.path.join(basedir,imagedir)):
        os.makedirs(os.path.join(basedir,imagedir), exist_ok=True)
    if not os.path.exists(os.path.join(basedir,npdir)):
        os.makedirs(os.path.join(basedir,npdir), exist_ok=True)

    for i, p in enumerate(poses):
        color, depth = imgs[i],depths[i]
        filename_color = '{}_Color'.format(i)
        filename_depth = '{}_Depth'.format(i)
        format_suffix = '.png'
        np.save(os.path.join(basedir, npdir, filename_color), color)
        np.save(os.path.join(basedir, npdir, filename_depth), depth)

        import matplotlib.pyplot as plt
        plt.imsave(os.path.join(basedir, imagedir, filename_color+format_suffix), color)

        frame_data = {}
        frame_data['file_path'] = os.path.join(imagedir, filename_color)
        frame_data['transform_matrix'] = p.tolist()
        frames.append(frame_data)

    json_object = json.dumps(tranforms, indent=2)
    with open(os.path.join(basedir, "transforms.json"), "w") as outfile:
        outfile.write(json_object)

"""
    return cam_to_world
"""
def gen_scene_obs_pose(T_scene_to_world):
    obs_pose = []
    for x_angle in np.linspace(-40,40,3):
        for y_angle in np.linspace(0,10,2):
            #镜头朝向桌面            
            T_cam_face_to_scene = np.eye(4) 
            T_cam_face_to_scene[:3, :3] = R.from_euler("xyz", [180, 0, -90], degrees=True).as_matrix()
            #相机位置升高
            T_trans_up_to_scene = np.eye(4)
            T_trans_up_to_scene[2,3] = 0.4
            T_cam_face_to_scene = T_trans_up_to_scene @ T_cam_face_to_scene
            #新观察位置
            T_obs_to_scene = np.eye(4)
            T_obs_to_scene[:3, :3] = R.from_euler("xyz", [x_angle, y_angle, 0], degrees=True).as_matrix()
            T_cam_face_to_scene = T_obs_to_scene @ T_cam_face_to_scene
            #转换为世界座标
            T_cam_face_to_world = T_scene_to_world @ T_cam_face_to_scene
            obs_pose.append(T_cam_face_to_world)
    return obs_pose

def visual_pose(ori, obs_pose):
    a = []
    for p in obs_pose:
        a.append(np.round(p @ np.array([0,0,0,1]), 3)[0:3])
        print(a)

    a = np.array(a)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(ori[0], ori[1], ori[1], marker='o')
    ax.scatter(a[...,0], a[...,1], a[..., 2])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7,0.7)
    ax.set_zlim(0,0.7)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def main():
    with open(os.path.join(os.path.dirname(__file__), "../config/cam_info_realsense.json"), 'r') as f:
        cam_info = json.load(f)
    cam_intr = np.array(cam_info["K"]).reshape(3, 3)
    T_cam_to_tool0 = np.array(cam_info["cam_to_tool0"]).reshape(4, 4)
    init_pose = config.init_position.tolist() + config.init_orientation.tolist()

    # init arm control
    arm = UR5Commander()
    print("Arm initialized")

    camera = RealSenseCommander()
    camera.start()
    print("Camera initialized")
    
    arm.set_pose(init_pose, wait=True)
    print('init pose')

    move_and_get_image(cam_info, arm, camera)

    arm.set_pose(init_pose, wait=True)
    print('end pose')

if __name__=="__main__":
    main()




from mimetypes import init
from ur5_commander import UR5Commander
import rospy
from configs import config
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import json
from matplotlib import pyplot as plt

import os
import numpy as np
import json
import matplotlib.pyplot as plt

def pose_to_transform_matrix(pose, T_cam_to_tool0):
    tool0_pose = pose
    T_tool0_to_world = np.eye(4)
    T_tool0_to_world[:3, :3] = R.from_quat(tool0_pose[3:]).as_matrix()
    T_tool0_to_world[:3, 3] = tool0_pose[:3]
    T_cam_to_world = T_tool0_to_world @ T_cam_to_tool0
    return T_cam_to_world

def gen_ur_pose(T_cam_to_world, T_cam_to_tool0):
    T_tool0_to_cam = np.linalg.inv(T_cam_to_tool0)
    T_tool0_to_world = T_cam_to_world @ T_tool0_to_cam
    tool0_pose = [] + T_tool0_to_world[:3,3].tolist() + R.from_matrix(T_tool0_to_world[:3,:3]).as_quat().tolist()
    return tool0_pose

def gen_ur_pose(T_):
    pose = [] + T_[:3,3].tolist() + R.from_matrix(T_[:3,:3]).as_quat().tolist()
    return pose

def move_arm():
    with open(os.path.join(os.path.dirname(__file__), "../config/cam_info_realsense.json"), 'r') as f:
        cam_info = json.load(f)
    cam_intr = np.array(cam_info["K"]).reshape(3, 3)
    T_cam_to_tool0 = np.array(cam_info["cam_to_tool0"]).reshape(4, 4)
    init_pose = config.init_position.tolist() + config.init_orientation.tolist()

    # init arm control
    arm = UR5Commander()
    rospy.loginfo("Arm initialized")
    arm.set_pose(init_pose, wait=True)
    print("set init pose")


    T_scene_to_world = np.eye(4)
    t_scene_to_world = [0.4, 0.05, 0.03]
    T_scene_to_world[:3, 3] = t_scene_to_world
    T_world_to_scene = np.linalg.inv(T_scene_to_world)

    #cam pose 
    obs_pose = []
    obs_pose.append(init_pose)

    for i, p in enumerate(obs_pose):
        arm.set_pose(p, wait=True)
        print(i)
        print(p)

def display():
    camera = get_camera()
    camera.start()
    print("Camera initialized")
    fig = plt.figure()

    while True:
        # get camera message
        color, depth = camera.get_image()
        plt.imshow(color)
        plt.show()

def move_and_get_image():
    with open(os.path.join(os.path.dirname(__file__), "../config/cam_info_realsense.json"), 'r') as f:
        cam_info = json.load(f)
    cam_intr = np.array(cam_info["K"]).reshape(3, 3)
    T_cam_to_tool0 = np.array(cam_info["cam_to_tool0"]).reshape(4, 4)
    init_pose = config.init_position.tolist() + config.init_orientation.tolist()

    # init arm control
    arm = UR5Commander()
    rospy.loginfo("Arm initialized")

    camera = get_camera()
    camera.start()
    print("Camera initialized")

    T_scene_to_world = np.eye(4)
    t_scene_to_world = [0.4, 0.05, 0.03]
    T_scene_to_world[:3, 3] = t_scene_to_world
    T_world_to_scene = np.linalg.inv(T_scene_to_world)

    #cam pose 
    obs_pose = []
    obs_pose.append(init_pose)

    #r1
    pose = [0.4326047628305254, -0.27601854575585716, 0.31502277104938686, -0.6403355586700344, 0.6720988786089179, 0.250780903513018, 0.2745221450239749]
    obs_pose.append(pose)
    #r2
    pose = [0.4312532598123046, -0.18189049230338578, 0.3598946475221223, -0.6653041189815552, 0.6934415638166361, 0.18048616670295264, 0.20960431881664293]
    obs_pose.append(pose)
    #u1
    pose = [0.6505597570576372, 0.023491647897313533, 0.3535222526998132, 0.662462771381004, -0.690616082823764, -0.2200022862901425, 0.18918640729838418]
    obs_pose.append(pose)
    #l1
    pose = [0.4851907067796292, 0.21611573307921111, 0.3947951168078107, 0.6711337567517277, -0.7033524657499528, 0.13911966930756467, 0.18846884910649842]
    obs_pose.append(pose)
    #l2
    pose = [0.4640954809151032, 0.29668814559290013, 0.2763665256365443, 0.6433902761197504, -0.6818904466956321, 0.22407260964788708, 0.2662063802867397]
    obs_pose.append(pose)
    #b1
    pose = [0.2939921356623896, 0.08324612624701633, 0.4019966841987688, -0.6597706741365947, 0.7273633243545239, -0.16364593286893767, 0.0941555127899283]
    obs_pose.append(pose)

    imgs = []
    depths = []
    poses = []
    for i, p in enumerate(obs_pose):
        arm.set_pose(p, wait=True)
        print(f'{i} {p}')
        color, depth = camera.get_image()
        print(f'color:{color}')
        print(f'depth:{depth}')
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
    tranforms['camera_angle_x'] = 0.9747124782401998
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

def get_transform_from_rotation(rot):
    t = np.eye(4)
    t[:3, :3] = rot
    return t

def gen_scene_obs_pose(T_scene_to_world):
    obs_pose = []
    for x_angle in np.linspace(-50,50,5):
        for y_angle in np.linspace(0,60,5):
            #镜头朝向桌面            
            T_cam_face_to_scene = np.eye(4) 
            T_cam_face_to_scene[:3, :3] = R.from_euler("xyz", [-180, 0, 90], degrees=True).as_matrix()
            #相机位置升高
            T_trans_up_to_scene = np.eye(4)
            T_trans_up_to_scene[2,3] = 0.5
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

def get_camera():
    pass

if __name__=="__main__":
    move_and_get_image()





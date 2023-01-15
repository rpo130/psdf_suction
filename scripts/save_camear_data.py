import os
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.pyplot as plt

def get_camera_pose():
    return np.eye(4)

def get_image():
    return np.zeros((200,200,3)), np.zeros((200,200))

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

if __name__ == '__main__':
    imgs = [get_image()[0],get_image()[0],get_image()[0]]
    depths = [get_image()[1],get_image()[1],get_image()[1]]
    poses = [get_camera_pose(),get_camera_pose(),get_camera_pose()]
    cam_intr = np.eye(3)
    gen_data(imgs,depths,poses, cam_intr)

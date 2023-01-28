from psdf import PSDF
from configs import config
import numpy as np

def test_fuse_point():
    psdf = PSDF(config.volume_shape, config.volume_resolution)
    psdf.fuse_point(np.array([0.0,-0.25,0.02]), config.T_world_to_volume)
    print(psdf.sdf == 0)
    print(np.where(psdf.sdf.cpu().numpy() == 0))
    point_map = flatten(psdf)
    point_img = point_map[..., 2] - config.T_volume_to_world[2, 3] / config.volume_range[2]
    import matplotlib.pyplot as plt
    plt.imshow(point_img)
    plt.show()

def test_fuse_contact():
    psdf = PSDF(config.volume_shape, config.volume_resolution)
    T = np.eye(4)
    T[:3,3] = [0.2,0.2,0.2]
    print(T)
    psdf.fuse_contact(T)
    print(psdf.sdf[(psdf.sdf > -1e-1) * (psdf.sdf < 1e-1)])
    point_map = flatten(psdf)
    point_img = point_map[..., 2] - config.T_volume_to_world[2, 3] / config.volume_range[2]
    import matplotlib.pyplot as plt
    plt.imshow(point_img)
    plt.show()

def test_fuse_depth():
    psdf = PSDF(config.volume_shape, config.volume_resolution, with_color=True)

    imgs, depths, poses, K = load_avt_data('/home/ai/codebase/nerf-pytorch/data/avt_data_glass_20230118_1/')
    # imgs, depths, poses, K = load_log()

    for i in range(len(poses)):
      T_cam_to_world = poses[i]
      T_cam_to_volume = config.T_world_to_volume @ T_cam_to_world
      psdf.fuse(depths[i], K, T_cam_to_volume, imgs[i])

    point_map, color_map = flatten(psdf)
    point_map = point_map @ config.T_volume_to_world[:3, :3].T + config.T_volume_to_world[:3, 3]

    point_img = point_map[..., 2] - config.T_volume_to_world[2, 3] / config.volume_range[2]
    import matplotlib.pyplot as plt
    plt.imshow(point_img)
    plt.colorbar()
    plt.show()

def flatten(psdf: PSDF, smooth=False, ksize=5, sigmaColor=0.1, sigmaSpace=5):
    import torch
    # find surface point
    #(250,250,250)
    surface_mask = psdf.sdf <= 0.01

    # get height map
    z_vol = torch.zeros_like(psdf.sdf).long()
    z_vol[surface_mask] = psdf.indices[surface_mask][:, 2]
    z_flat = torch.max(z_vol, dim=-1)[0]
    #(250,250)
    height_map = psdf.positions[..., 2].take(z_flat)
    if smooth:
        import cv2
        height_map = cv2.bilateralFilter(height_map, ksize, sigmaColor, sigmaSpace)

    # get point map
    # (250,250,3)
    point_map = psdf.positions[:, :, 0, :].clone()
    point_map[..., 2] = height_map

    # get color map
    color_map = psdf.rgb.take(z_flat)

    return (point_map.cpu().numpy(), 
            color_map.cpu().numpy())

import os,json
import json
from scipy.spatial.transform import Rotation as R
def load_avt_data(basedir):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    all_imgs = []
    all_depths= []
    all_poses = []
    imgs = []
    poses = []
    depths = []
        
    for frame in meta['frames'][::1]:
        file_path = frame['file_path']
        file_path = file_path.replace('images', 'np')
        fname = os.path.join(basedir, file_path+".npy")
        imgs.append(np.load(fname))
        file_path = file_path.replace('Color', 'Depth')
        fname = os.path.join(basedir, file_path+".npy")
        depths.append(np.load(fname))
        T_cam_to_world = np.array(frame['transform_matrix'])
        poses.append(T_cam_to_world)
    poses = np.array(poses).astype(np.float32)
    
    all_imgs.append(imgs)
    all_poses.append(poses)
    all_depths.append(depths)

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    depths = np.concatenate(all_depths, 0)
    
    imgs = imgs[...,:3]
        
    fx = meta['fx']
    fy = meta['fy']
    cx = meta['cx']
    cy = meta['cy']

    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    return imgs,depths, poses, K

import matplotlib.image as mimg

def load_log():
  data_basedir = '/home/ai/codebase/nerf-pytorch/data/avt_data_glass_20230118_1'
  log_basedir = '/home/ai/codebase/nerf-pytorch/logs/avt_data_test/renderonly_test_739999'

  with open(os.path.join(data_basedir, 'transforms.json'), 'r') as fp:
    meta = json.load(fp)

  imgs = []
  depths = []
  poses = []

  fx = meta['fx']
  fy = meta['fy']
  cx = meta['cx']
  cy = meta['cy']

  K = np.array([
      [fx, 0, cx],
      [0, fy, cy],
      [0, 0, 1]])

  with open(os.path.join(log_basedir, '..', 'split.txt')) as fp:
    for line in fp:
      if 'test' in line:
        line = line[7:].strip()
        file_seq = line.split(",")
        for i in range(len(file_seq)):
          poses.append(np.array(meta['frames'][int(file_seq[i])]['transform_matrix']))
          imgs.append(mimg.imread(os.path.join(log_basedir, '{:03d}.png'.format(i))).astype(np.float32))
          depths.append(np.load(os.path.join(log_basedir, '{:03d}_depth_ff.npy'.format(i))).astype(np.float32))

        break
  return imgs, depths, poses, K

if __name__ == "__main__":
    test_fuse_depth()
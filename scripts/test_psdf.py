from psdf import PSDF
from configs import config
import numpy as np
import matplotlib.pyplot as plt
from analyser.vacuum_cup_analyser import VacuumCupAnalyser
from node_planner import compute_score_inner

def test_fuse_point():
    psdf = PSDF(config.volume_shape, config.volume_resolution)
    psdf.fuse_point(np.array([0.0,-0.25,0.02]), config.T_world_to_volume)
    print(psdf.sdf == 0)
    print(np.where(psdf.sdf.cpu().numpy() == 0))
    point_map = psdf.flatten()
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
    point_map = psdf.flatten()
    point_img = point_map[..., 2] - config.T_volume_to_world[2, 3] / config.volume_range[2]
    import matplotlib.pyplot as plt
    plt.imshow(point_img)
    plt.show()

def test_fuse_depth():
    psdf = PSDF(config.volume_shape, config.volume_resolution, with_color=True)

    imgs, depths, poses, K = load_avt_data('/home/ai/codebase/nerf-pytorch/data/avt_data_glass_20230118_1/')
    # imgs, depths, poses, K = load_log()

    for i in range(0,len(poses)):
      T_cam_to_world = poses[i]
      T_cam_to_volume = config.T_world_to_volume @ T_cam_to_world
      psdf.fuse(depths[i], K, T_cam_to_volume, imgs[i])

    point_map, color_map, *_ = psdf.flatten()
    point_map = point_map @ config.T_volume_to_world[:3, :3].T + config.T_volume_to_world[:3, 3]

    point_img = point_map[..., 2] - config.T_volume_to_world[2, 3] / config.volume_range[2]

    v,f = psdf.get_point_cloud(config.T_volume_to_world)
    o3d_display_point(v)
    mp_display_point(v)

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


def mp_display_point(verts):
  plt.style.use('_mpl-gallery')
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  x,y,z = verts[...,0],verts[...,1],verts[...,2]
  sample = 10
  ax.scatter(x[::sample], y[::sample], z[::sample])

  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  plt.show()	

def o3d_display_point(verts):
  import open3d

  pcd = open3d.geometry.PointCloud()

  pcd.points = open3d.utility.Vector3dVector(verts)
  mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
  open3d.visualization.draw_geometries([pcd, mesh_frame])

DEVICE = "cpu"

def test_grasp():
    # init analyser
    analyser = VacuumCupAnalyser(radius=config.gripper_radius,
                                 height=config.gripper_height,
                                 num_vertices=config.gripper_vertices,
                                 angle_threshold=config.gripper_angle_threshold)
    print("Analyser initialized")

    psdf = PSDF(config.volume_shape, config.volume_resolution,
                device=DEVICE, with_color=True)
    print("PSDF initialized")


    import cv2
    # fpath = '/home/ai/codebase/instant-ngp/data/nerf/dex_nerf_simulated_wineglass_pose02/depth_dex/r_7.png'
    fpath = '/home/ai/codebase/instant-ngp/data/nerf/dex_nerf_real_flask/depth_dex/14_Color.png'
    depth = cv2.imread(fpath, cv2.IMREAD_ANYDEPTH)
    depth = depth.astype(np.float32)
    depth = depth / 1000

    T_cam_to_world = np.array([
        [
          -0.35324800774031584,
          -0.8585629231500284,
          0.371598552801859,
          0.2741426251726115
        ],
        [
          0.9355296981475072,
          -0.32415642513544335,
          0.1403809636789037,
          0.19120277121291676
        ],
        [
          -6.941215028930547e-05,
          0.39723079555540103,
          0.9177185917385781,
          0.5915793498273141
        ],
        [
          0.0,
          0.0,
          0.0,
          1.0
        ]
      ])

    T_img_to_cam_face = np.eye(4)
    T_img_to_cam_face[:3,:3] = R.from_euler("xyz", [180,0,0], degrees=True).as_matrix()
    T_cam_to_world = T_cam_to_world @ T_img_to_cam_face 
    print(T_cam_to_world)

    T_cam_to_volume = config.T_world_to_volume @ T_cam_to_world


    camera_angle_x = 1.2303659303290422 #0.6911112070083618
    H,W = depth.shape
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])

    psdf.fuse(np.copy(depth), K, T_cam_to_volume, color=None)

    o3d_display_point(psdf.get_point_cloud()[0])

    # for i, p in enumerate(render_poses):
    #     T_cam_to_world = p
    #     T_cam_to_volume = config.T_world_to_volume @ T_cam_to_world
    #     # fuse new data to psdf
    #     psdf.fuse(np.copy(depths[i]), K, T_cam_to_volume, color=np.copy(colors[i]))

    # flatten to 2D point image
    point_map, normal_map, variance_map, _ = psdf.flatten(smooth=True)
    point_map = point_map @ config.T_volume_to_world[:3,:3].T + config.T_volume_to_world[:3, 3]

    # analysis
    normal_mask = normal_map[..., 2] > np.cos(config.gripper_angle_threshold/180*np.pi)
    variance_mask = variance_map < 1e-2
    z_mask = point_map[:, :, 2] > -0.1
    final_mask = normal_mask * variance_mask #* z_mask
    obj_ids = np.where(final_mask != 0)
    vision_dict = {"point_cloud": point_map,
                    "normal": -normal_map}
    graspable_map = analyser.analyse(vision_dict, obj_ids).astype(np.float32)

    # update grasp pose
    grasp_position = np.array([0,0,0])
    score = compute_score_inner(graspable_map, point_map,
                            normal_map, grasp_position)
    idx = np.argmax(score)
    i, j = idx // config.volume_shape[0], idx % config.volume_shape[1]
    grasp_position = point_map[i, j]
    grasp_normal = normal_map[i, j]

    fig = plt.figure()
    ax = fig.add_subplot(231)
    ax.imshow(depth)
    ax.set_title('input depth')

    ax = fig.add_subplot(232)
    ax.imshow(point_map[..., 2])
    ax.set_title('height map')

    ax = fig.add_subplot(233)
    ax.imshow(graspable_map)
    ax.set_title('grasp map')
    
    ax = fig.add_subplot(235)
    ax.imshow(point_map[..., 2])
    ax.scatter(i, j, s=50, c='red', marker='o')
    ax.set_title('height map with point mark')

    ax = fig.add_subplot(236)
    ax.imshow(graspable_map)
    ax.scatter(i, j, s=50, c='red', marker='o')
    ax.set_title('grasp map with point mark')

    print(f'grasp position {grasp_position}, normal {grasp_normal}')
    plt.show()

if __name__ == "__main__":
    # test_fuse_depth()
    test_grasp()
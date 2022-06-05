import os

import numpy as np
import torch
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule
    print('Import pycuda success!')
except Exception as err:
    print('Warning: {}'.format(err))
    print('Failed to import PyCUDA.')
    exit()


# Warning:do not delete!This just prepares for interaction between torch and cuda.
x = torch.cuda.FloatTensor(8)


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


class VacuumCupAnalyser(object):
    def __init__(self, radius=0.01, height=0.04, num_vertices=8, angle_threshold=np.pi / 4):
        """
        Construct a mass-spring model for vacuum gripper.
        The apex is set to be the origin of the vacuum gripper frame.
        :param radius: The radius of vacuum cup base. The unit is meter.
        :param height: The distance between apex and base. The unit is meter.
        :param num_vertices: The number of vertices sampled from the edge of base.
        :param angle_threshold: The maximum incline angle for the gripper, represented in radius.
        """
        self.radius = radius
        self.height = height
        self.num_vertices = num_vertices
        self.threshold = angle_threshold
        self.apex, self.base_center, self.vertices = self.get_vacuum_gripper_model()
        self.natural_perimeter = np.linalg.norm(self.vertices[:, 0] - self.vertices[:, 1])
        self.natural_cone = np.linalg.norm(self.apex - self.vertices[:, 1])
        self.natural_flexion = np.linalg.norm(self.vertices[:, 0] - self.vertices[:, 2])

        self.ctx = cuda.Device(0).make_context()
        c_file = open(os.path.join(os.path.dirname(__file__), "vacuum_cup_analyser.cpp"), 'r')
        c_string = c_file.read()
        self._cuda_src_mod = SourceModule(c_string)
        self._cuda_extract = self._cuda_src_mod.get_function("extract")

        self._threads_per_block = 256
        self._grid_dim_x = 740
        self._n_gpu_loops = 0

    def analyse(self, vision_dict, obj_ids, half_patch=15):

        height, width = vision_dict['point_cloud'].shape[:2]
        point_cloud = (vision_dict['point_cloud'].reshape(-1)).astype(np.float32)
        normal = (vision_dict['normal'].reshape(-1)).astype(np.float32)
        centerx = obj_ids[0].astype(np.float32)
        centery = obj_ids[1].astype(np.float32)

        len = np.zeros(500000).astype(np.float32)
        len_gpu = cuda.mem_alloc(len.nbytes)
        label_mask = np.zeros(height * width).astype(np.float32)
        label_mask_gpu = cuda.mem_alloc(label_mask.nbytes)
        cuda.memcpy_htod(label_mask_gpu, label_mask)

        print('Points to be analaysed: {}'.format(np.shape(obj_ids[0])[0]))
        grid_dim_x = int(np.ceil(np.shape(obj_ids[0])[0] / float(self._threads_per_block)))
        n_gpu_loops = np.ceil((np.shape(obj_ids[0])[0] / np.float(
            grid_dim_x * self._threads_per_block))).astype(np.int)
        patch_size = half_patch*2+1
        pc_buffer = np.zeros((obj_ids[0].shape[0], patch_size, patch_size, 3)).astype(np.float32)
        dist = np.zeros((obj_ids[0].shape[0], patch_size, patch_size)).astype(np.float32)

        for gpu_loop_idx in range(n_gpu_loops):
            self._cuda_extract(cuda.In(point_cloud), cuda.In(normal),
                                cuda.In(centerx), cuda.In(centery), label_mask_gpu,
                                cuda.In(self.apex),
                                cuda.In(self.base_center),
                                cuda.In(np.reshape(self.vertices, -1)),
                                cuda.In(pc_buffer),
                                cuda.In(dist),
                                cuda.In(np.asarray([
                                    gpu_loop_idx,
                                    height,
                                    width,
                                    np.shape(obj_ids[0])[0],
                                    half_patch,
                                    self.natural_perimeter,
                                    self.natural_flexion,
                                    self.natural_cone], np.float32)),
                                len_gpu,
                                block=(self._threads_per_block, 1, 1),
                                grid=(grid_dim_x, 1)
                                )
        cuda.memcpy_dtoh(len, len_gpu)
        cuda.memcpy_dtoh(label_mask, label_mask_gpu)
        print('Points remained after getting patch : {}'.format(np.sum(len)))
        print('Grasp points : {}'.format(np.sum(label_mask)))
        label_mask = (label_mask.astype(np.uint8)).reshape((height, width))
        return label_mask

    def get_vacuum_gripper_model(self):
        apex = np.zeros([3], dtype=np.float32)
        base_center = np.array([0.0, 0.0, self.height], dtype=np.float32)
        vertices = []
        for i in range(self.num_vertices):
            curr_angle = np.pi * 2 * i / self.num_vertices
            vertices.append([np.sin(curr_angle) * self.radius, np.cos(curr_angle) * self.radius, self.height])
        vertices = np.array(vertices, dtype=np.float32)
        return apex, base_center, vertices.T

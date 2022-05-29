import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    FUSION_GPU_MODE = 1
    print('Import pycuda success!')
except Exception as err:
    print('Warning: {}'.format(err))
    print('Failed to import PyCUDA. Running fusion in CPU mode.')
    FUSION_GPU_MODE = 0

# Warning:do not delete!This just prepares for interaction between torch and cuda.
x = torch.cuda.FloatTensor(8)


class Holder(pycuda.driver.PointerHolderBase):
    def __init__(self, t):
        super(Holder, self).__init__()
        self.t = t
        self.gpudata = t.data_ptr()

    def get_pointer(self):
        return self.t.data_ptr()


class VacuumGripper(object):
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

        # Cuda kernel function (C++)
        self._cuda_src_mod = SourceModule("""
        #include <math.h>
        #include <stdio.h>
        #include <stdbool.h>
        __device__ void quicksort(float * a,int * indices,int left,int right) 
        {
            int i,j,t,temp2;
            float temp1;
            if(left>right)
                return;

            temp1=a[left];
            temp2=indices[left];

            i=left;
            j=right;
            while(i!=j) 
            {
                while(a[j]>=temp1&&j>i)
                    j--;
                while(a[i]<=temp1&&j>i)
                    i++;
                if(i<j) 
                {
                    t=a[i];
                    a[i]=a[j];
                    a[j]=t;
                    t=indices[i];
                    indices[i]=indices[j];
                    indices[j]=t;
                }
            }
            a[left]=a[i];
            a[i]=temp1;
            indices[left]=indices[i];
            indices[i]=temp2;
            quicksort(a,indices,left,i-1);
            quicksort(a,indices,i+1,right);

        }
        __device__ int fact(int f,float * a)
        {
          if (f == 0)
            return 1+(int)a[0];
          else
            return f * fact(f - 1,a);
        }
        __device__ float sign(float p1[3], float p2[3], float p3[3])
        {
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1]);
        }
        __device__ bool is_in_triangle(float p1[3], float p2[3], float p3[3])
        {
            float origin[3]={0};
            float d1 = sign(origin, p1, p2);
            float d2 = sign(origin, p2, p3);
            float d3 = sign(origin, p3, p1);

            bool has_neg = (d1 < 0) || (d2 < 0) || (d3 < 0);
            bool has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0);
            return !(has_neg && has_pos);
        }
        __device__ float get_interpolation_z(float point[3],float p1[3],float p2[3],float p3[3])
        {
            float A = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]);
            float B = (p3[0] - p1[0]) * (p2[2] - p1[2]) - (p2[0] - p1[0]) * (p3[2] - p1[2]);
            float C = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]);
            float D = -(A * p1[0] + B * p1[1] + C * p1[2]);
            return -(A * point[0] + B * point[1] + D) / C;
        }
        __device__ void sort (float * arr, int * indices,int n)
        {
            float temp = 0;
            int i,j;
            for(i = 0;i<n;i++)
            {
                temp = 100000;
                indices[i] = 77; 
                for(j=0;j<31*31;j++)
                {
                    if(arr[j]<temp)
                    {
                        temp = arr[j];
                        indices[i] = j;
                    }
                }
                arr[indices[i]] = 100000;
            }

        }
        __device__ bool 


        __device__ int my_f(int x, int y)
        { 
            return x+y;
        }
        



        """)

        self._cuda_extract = self._cuda_src_mod.get_function("extract")
        gpu_dev = cuda.Device(1)
        self._threads_per_block = 256
        self._grid_dim_x = 740
        self._n_gpu_loops = 0

    def get_triangle_point(float * point,float * interpolated_point,float (*point_cloud)[3])
        {
            float dist[31*31]={0};
            int indices[20]={0};
            int count=0,i=0,j=1,k=2;
            for(i = 0;i<31*31;i++)
            {
                dist[i]=pow((point_cloud[i][0]-point[0]),2)+pow((point_cloud[i][1]-point[1]),2);
            }
            sort(dist,indices,20);
            bool find_triangle = false;
            float p1[3],p2[3],p3[3];
            for(i=0;i<10;i++)
            {
                for(j=i+1;j<10;j++)
                {
                    for(k=j+1;k<20;k++)
                    {
                        count++;
                        for(int l=0;l<3;l++)
                        {
                            p1[l] = point_cloud[indices[i]][l]-point[l];
                            p2[l] = point_cloud[indices[j]][l]-point[l];
                            p3[l] = point_cloud[indices[k]][l]-point[l];
                        }
                        find_triangle = is_in_triangle(p1,p2,p3);
                        if((find_triangle) || count>100)
                            break;
                    }
                    if((find_triangle) || count>100)
                        break;
                }
                if((find_triangle) || count>100)
                    break;
            }
            if(find_triangle)
            {
                for(i=0;i<3;i++)
                {
                    p1[i] += point[i];
                    p2[i] += point[i];
                    p3[i] += point[i];
                }
                interpolated_point[0] = point[0];
                interpolated_point[1] = point[1];
                interpolated_point[2] = get_interpolation_z(point,p1,p2,p3);
            }

            return find_triangle;

        }

    def extract(self, point_cloud, normal, centerx, centery, label_mask, apex, base_center, vertices, other_params, len_test):

            //int gpu_loop_idx = (int) other_params[0];

            height = other_params[1]
            width = other_params[2]
            len = other_params[3]
            half_patch = other_params[4]
            natural_perimeter = other_params[5]
            natural_flexion = other_params[6]
            natural_cone = other_params[7]


            int idx = (int)(blockIdx.x*blockDim.x+threadIdx.x);
            if (idx>=len)
                return;

            int id_x = (int) centerx[idx];
            int id_y = (int) centery[idx];

            if ((id_x - half_patch < 0) || (id_x + half_patch >= height) || (id_y - half_patch < 0) || (id_y + half_patch >= width))
                return;


            len_test[idx] = len_test[idx]+1;


            s = 15
            EPSILON = 1e-6
            pc_patch = point_cloud[id_x-s:id_x+s+1, id_y-s:id_y+s+1, :]
            grasp_direction = -normal[id_x][id_y]
            grasp_direction = grasp_direction / (np.linalg.norm(grasp_direction) + EPSILON)
            grasp_position = point_cloud[id_x, id_y]

            #get_transformation
            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            r_axis = np.cross(z_axis, grasp_direction)
            r_axis = r_axis / (np.linalg.norm(r_axis)+EPSILON)
            skew_symmetric_matrix = np.array([[0.0, -r_axis[2], r_axis[1]],
                                              [r_axis[2], 0.0, -r_axis[0]],
                                              [-r_axis[1], r_axis[0], 0.0]], dtype=np.float32)
            r_axis = np.expand_dims(r_axis, axis=1)
            cos = z_axis @ grasp_direction
            sin = np.sin(np.arccos(cos))
            r_mat = cos * np.identity(3, dtype=np.float32) + \
                    (1 - cos) * r_axis[:, None] * r_axis[None, :] + \
                    sin * skew_symmetric_matrix
            T_g2w = np.concatenate([r_mat, grasp_position[:, None]], axis=1)

            pc_g = pc_patch @ T_g2w[:3, :3] - T_g2w[:3, 3] @ T_g2w[:3, :3]
            for i in range(num_vertices):
                self.get_trangle_points(self.vertices[i], pc_g)


            r_axis = np.cross(grasp_direction, z_axis)
            r_axis = r_axis/(np.linalg.norm(r_axis)+EPSILON)

            cos_ = temp1
            sin_ = np.sin(np.arccos(cos_))

            transformation = np.eye(4)
            r_mat = np.array(
                [
                    [cos_+(1-cos_)*r_axis[0]**2, (1-cos_)*r_axis[0]*r_axis[1]-sin_*r_axis[2], (1-cos_)*r_axis[0]*r_axis[2]-sin_*r_axis[1]],
                    [(1-cos_)*r_axis[1]*r_axis[0]-sin_*r_axis[2], cos_+(1-cos_)*r_axis[1]**2, (1-cos_)*r_axis[1]*r_axis[2]-sin_*r_axis[0]],
                    [(1-cos_)*r_axis[2]*r_axis[0]-sin_*r_axis[1], (1-cos_)*r_axis[2]*r_axis[1]-sin_*r_axis[0], cos_+(1-cos_)*r_axis[2]**2],
                ]
            )
            t = grasp_center - grasp_center @ r_mat.T
            transformed_apex = apex @ r_mat.T + t
            transformed_base_center = base_center @ r_mat.T + t


            float transformation_w2g[3][4] = {0};
            float inverse_apex[3]={0};
            for (i = 0;i<3;i++)
            {
                for (j = 0;j<3;j++)
                {
                    inverse_apex[i] -= transformation[j][i]*transformed_apex[j];
                }
            }
            // inverse_apex checked ok

            float transformed_pc[31*31][3]={0};
            for (i = 0;i<31*31;i++)
            {
                for (j = 0;j<3;j++)
                {
                     for(k = 0 ;k<3;k++)
                     {
                         transformed_pc[i][j] += pc_patch[i][k]*transformation[k][j];
                     }
                     transformed_pc[i][j] += 1*inverse_apex[j];
                }
            }
            //transformed pc checked ok!
            float project_vertices[8][3]={0};
            float interpolated_point[3]={0};
            float perimeter[40][3]={0};
            float flexion[40][3]={0};
            float length_of_cones[8]={0};
            float length_of_perimeters[8]={0};
            float length_of_flexions[8]={0};
            float point[3]={0};
            bool find_triangle = true;
            for(i=0;i<8;i++)
            {
                point[0] = vertices[i];
                point[1] = vertices[i+8];
                point[2] = vertices[i+16];
                find_triangle = get_triangle_point(point,interpolated_point,transformed_pc);
                if(!find_triangle)
                    return;
                project_vertices[i][0] = interpolated_point[0];
                project_vertices[i][1] = interpolated_point[1];
                project_vertices[i][2] = interpolated_point[2];
                perimeter[i*5][0] = interpolated_point[0];
                perimeter[i*5][1] = interpolated_point[1];
                perimeter[i*5][2] = interpolated_point[2];
                flexion[i*5][0] = interpolated_point[0];
                flexion[i*5][1] = interpolated_point[1];
                flexion[i*5][2] = interpolated_point[2];
            }
            threshold = 0.1;

            // everything checked
            for(i=0;i<8;i++)
            {
                // compute length of cones
                length_of_cones[i] = sqrt(pow(project_vertices[i][0]-apex[0],2)+\
                                     pow(project_vertices[i][1]-apex[1],2)+\
                                     pow(project_vertices[i][2]-apex[2],2));

                if((length_of_cones[i]<(1-threshold)*natural_cone) || (length_of_cones[i]>(1+threshold)*natural_cone))
                    return;

                // compute length of perimeters by interpolation
                // notice: keep vertice points unchanged (i*5 in perimeter)
                for(j=1;j<5;j++)
                {
                    // sample a point from the perimeter
                    temp1 = (float)j / 5.0;
                    point[0] = (1 - temp1) * vertices[i] + temp1 * vertices[(i+1)%8];
                    point[1] = (1 - temp1) * vertices[i+8] + temp1 * vertices[(i+1)%8+8];
                    point[2] = (1 - temp1) * vertices[i+16] + temp1 * vertices[(i+1)%8+16];

                    find_triangle = get_triangle_point(point,interpolated_point,transformed_pc);
                    if(!find_triangle)
                        return;
                    perimeter[i*5+j][0] = interpolated_point[0];
                    perimeter[i*5+j][1] = interpolated_point[1];
                    perimeter[i*5+j][2] = interpolated_point[2];
                    length_of_perimeters[i] += sqrt(pow(interpolated_point[0]-perimeter[i*5+j-1][0],2)+\
                                         pow(interpolated_point[1]-perimeter[i*5+j-1][1],2)+\
                                         pow(interpolated_point[2]-perimeter[i*5+j-1][2],2));
                }
                length_of_perimeters[i] += sqrt(pow(perimeter[(i+1)*5%40][0]-perimeter[i*5+4][0],2)+\
                                     pow(perimeter[(i+1)*5%40][1]-perimeter[i*5+4][1],2)+\
                                     pow(perimeter[(i+1)*5%40][2]-perimeter[i*5+4][2],2));

                if((length_of_perimeters[i]<(1-threshold)*natural_perimeter) || (length_of_perimeters[i]>(1+threshold)*natural_perimeter))
                    return;


                // compute length of flexions by interpolation
                // notice: keep vertice points unchanged (i*5 in flexion)
                for(j=1;j<5;j++)
                {
                    // sample a point from the perimeter
                    temp2 = (float)j / 5.0;
                    point[0] = (1 - temp2) * vertices[i] + temp2 * vertices[(i+2)%8];
                    point[1] = (1 - temp2) * vertices[i+8] + temp2 * vertices[(i+2)%8+8];
                    point[2] = (1 - temp2) * vertices[i+16] + temp2 * vertices[(i+2)%8+16];
                    find_triangle = get_triangle_point(point,interpolated_point,transformed_pc);
                    if(!find_triangle)
                        return;
                    flexion[i*5+j][0] = interpolated_point[0];
                    flexion[i*5+j][1] = interpolated_point[1];
                    flexion[i*5+j][2] = interpolated_point[2];
                    length_of_flexions[i] += sqrt(pow(interpolated_point[0]-flexion[i*5+j-1][0],2)+\
                                         pow(interpolated_point[1]-flexion[i*5+j-1][1],2)+\
                                         pow(interpolated_point[2]-flexion[i*5+j-1][2],2));
                }
                length_of_flexions[i] += sqrt(pow(flexion[(i+2)*5%40][0]-flexion[i*5+j-1][0],2)+\
                                     pow(flexion[(i+2)*5%40][1]-flexion[i*5+j-1][1],2)+\
                                     pow(flexion[(i+2)*5%40][2]-flexion[i*5+j-1][2],2));

                if((length_of_flexions[i]<(1-threshold)*natural_flexion) || (length_of_flexions[i]>(1+threshold)*natural_flexion))
                    return;

            }

            label_mask[id_x*width+id_y] = label_mask[id_x*width+id_y]+1;

        }

    def update_visiondict(self, vision_dict, obj_ids, half_patch=15, use_gpu=True):
        gpu_mode = use_gpu and FUSION_GPU_MODE
        if gpu_mode:
            height, width, _ = vision_dict['point_cloud'].shape
            point_cloud = (vision_dict['point_cloud'].reshape(-1)).astype(np.float32)
            normal = (vision_dict['normal'].reshape(-1)).astype(np.float32)
            centerx = obj_ids[0].astype(np.float32)
            centery = obj_ids[1].astype(np.float32)

            len = np.zeros(500000).astype(np.float32)
            len_gpu = cuda.mem_alloc(len.nbytes)
            label_mask = np.zeros(height * width).astype(np.float32)
            label_mask_gpu = cuda.mem_alloc(label_mask.nbytes)

            print('Points to be analaysed: {}'.format(np.shape(obj_ids[0])[0]))
            grid_dim_x = int(np.ceil(np.shape(obj_ids[0])[0] / float(self._threads_per_block)))
            n_gpu_loops = np.ceil((np.shape(obj_ids[0])[0] / np.float(
                grid_dim_x * self._threads_per_block))).astype(np.int)

            for gpu_loop_idx in range(n_gpu_loops):
                self._cuda_extract(cuda.In(point_cloud), cuda.In(normal),
                                   cuda.In(centerx), cuda.In(centery), label_mask_gpu,
                                   cuda.In(self.apex),
                                   cuda.In(self.base_center),
                                   cuda.In(np.reshape(self.vertices, -1)),
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
            return label_mask, gpu_mode

    def get_vacuum_gripper_model(self):
        apex = np.zeros([3], dtype=np.float32)
        base_center = np.array([0.0, 0.0, self.height], dtype=np.float32)
        vertices = []
        for i in range(self.num_vertices):
            curr_angle = np.pi * 2 * i / self.num_vertices
            vertices.append([np.sin(curr_angle) * self.radius, np.cos(curr_angle) * self.radius, self.height])
        vertices = np.array(vertices, dtype=np.float32)
        return apex, base_center, vertices.T

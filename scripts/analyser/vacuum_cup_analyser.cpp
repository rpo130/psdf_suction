#include <math.h>
#include <stdio.h>
#include <stdbool.h>

#include "cuda_runtime_api.h"

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
__device__ void sort (float * arr, int * indices,int n,int patch_size)
{
    float temp = 0;
    int i,j;
    for(i = 0;i<n;i++)
    {
        temp = 100000;
        indices[i] = 77;
        for(j=0;j<patch_size*patch_size;j++)
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
__device__ bool get_triangle_point(float * point,float * interpolated_point,float (*point_cloud)[3], int patch_size, float * dist)
{
    int indices[20]={0};
    int count=0,i=0,j=1,k=2;
    for(i = 0;i<patch_size*patch_size;i++)
    {
        dist[i]=pow((point_cloud[i][0]-point[0]),2)+pow((point_cloud[i][1]-point[1]),2);
    }
    sort(dist,indices,20,patch_size);
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


__device__ int my_f(int x, int y)
{
    return x+y;
}
__global__ void extract(float * point_cloud,
                    float * normal,
                    float * centerx,
                    float * centery,
                    float * label_mask,
                    float * apex,
                    float * base_center,
                    float * vertices,
                    float * pc_buffer,
                    float * dist,
                    float * other_params,
                    float * len_test)
{
    //int gpu_loop_idx = (int) other_params[0];
    int height = (int) other_params[1];
    int width = (int) other_params[2];
    int len = (int) other_params[3];
    int half_patch = (int) other_params[4];
    float natural_perimeter = other_params[5];
    float natural_flexion = other_params[6];
    float natural_cone = other_params[7];

    int idx = (int)(blockIdx.x*blockDim.x+threadIdx.x);
    if (idx>=len)
        return;

    int id_x = (int) centerx[idx];
    int id_y = (int) centery[idx];
    int patch_size = half_patch*2+1;
    float * this_pc_buffer = pc_buffer + (blockIdx.x*blockDim.x+threadIdx.x)*patch_size*patch_size*3;
    float * this_dist = dist + (blockIdx.x*blockDim.x+threadIdx.x)*patch_size*patch_size;


    if ((id_x - half_patch < 0) || (id_x + half_patch >= height) || (id_y - half_patch < 0) || (id_y + half_patch >= width))
        return;
    len_test[idx] = len_test[idx]+1;

    int i=0,j=0,k=0,x=0,y=0;
    for(x = id_x-half_patch;x<=id_x+half_patch;x++)
    {
        for(y = id_y-half_patch; y<=id_y+half_patch;y++)
        {
            *(this_pc_buffer+i*3+0) = point_cloud[(x*width+y)*3+0];
            *(this_pc_buffer+i*3+1) = point_cloud[(x*width+y)*3+1];
            *(this_pc_buffer+i*3+2) = point_cloud[(x*width+y)*3+2];
            i++;
        }
    }
    float grasp_direction[3];
    float grasp_center[3];
    float temp1 = 0;
    float temp2 = 0;
    float z_axis[3] = {0,0,-1};

    for(i=0;i<3;i++)
    {
        grasp_direction[i] = normal[(id_x*width+id_y)*3+i];
        grasp_center[i] = point_cloud[(id_x*width+id_y)*3+i];
        temp1 += grasp_direction[i]*z_axis[i];
    }

    float threshold = 3.14159265358979323846/4.0;
    if ((grasp_direction[2]>=0) || (temp1<cos(threshold))){
        return;
    }

    //transform_vacuum_gripper(grasp_point=grasp_center, surface_normal=grasp_direction)
    //get_transformation
    float rotation_axis[3];
    // cross dot
    rotation_axis[0] = grasp_direction[1]*(-z_axis[2])-grasp_direction[2]*z_axis[1];
    rotation_axis[1] = grasp_direction[2]*z_axis[0]-grasp_direction[0]*(-z_axis[2]);
    rotation_axis[2] = grasp_direction[0]*z_axis[1]-grasp_direction[1]*z_axis[0];
    float norm = sqrt(pow(rotation_axis[0],2)+pow(rotation_axis[1],2)+pow(rotation_axis[2],2));
    if (norm>0)
    {
        rotation_axis[0] = rotation_axis[0]/norm;
        rotation_axis[1] = rotation_axis[1]/norm;
        rotation_axis[2] = rotation_axis[2]/norm;
    }
    float cos_ = -grasp_direction[0]*z_axis[0]-grasp_direction[1]*z_axis[1]+grasp_direction[2]*z_axis[2];
    float sin_ = sin(acos(cos_));
    float transformation[3][4]={0};
    transformation[0][0] = cos_+(1-cos_)*rotation_axis[0]*rotation_axis[0];
    transformation[0][1] = (1-cos_)*rotation_axis[0]*rotation_axis[1]-sin_*rotation_axis[2];
    transformation[0][2] = (1-cos_)*rotation_axis[0]*rotation_axis[2]+sin_*rotation_axis[1];

    transformation[1][0] = (1-cos_)*rotation_axis[1]*rotation_axis[0]+sin_*rotation_axis[2];
    transformation[1][1] = cos_+(1-cos_)*rotation_axis[1]*rotation_axis[1];
    transformation[1][2] = (1-cos_)*rotation_axis[1]*rotation_axis[2]-sin_*rotation_axis[0];

    transformation[2][0] = (1-cos_)*rotation_axis[2]*rotation_axis[0]-sin_*rotation_axis[1];
    transformation[2][1] = (1-cos_)*rotation_axis[2]*rotation_axis[1]+sin_*rotation_axis[0];
    transformation[2][2] = cos_+(1-cos_)*rotation_axis[2]*rotation_axis[2];

    for (i = 0;i<3;i++)
    {
        temp1 = 0;
        for (int j = 0;j<3;j++)
        {
            temp1 += transformation[i][j]*base_center[j];
        }
        transformation[i][3] = grasp_center[i]-temp1;
    }
    float transformed_apex[3];
    float transformed_base_center[3];
    for (i = 0;i<3;i++)
    {
        temp1 = 0;
        temp2 = 0;
        for (j = 0;j<3;j++)
        {
            temp1 += transformation[i][j]*apex[j];
            temp2 += transformation[i][j]*base_center[j];
        }
        transformed_apex[i] =  transformation[i][3]+temp1;
        transformed_base_center[i]  = transformation[i][3]+temp2;
    }
    // checked ok

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

    for (i = 0;i<patch_size*patch_size;i++)
    {
        float tmpx=0;
        float tmpy=0;
        float tmpz=0;
        for(k = 0 ;k<3;k++)
        {
            tmpx += *(this_pc_buffer+i*3+k)*transformation[k][0];
            tmpy += *(this_pc_buffer+i*3+k)*transformation[k][1];
            tmpz += *(this_pc_buffer+i*3+k)*transformation[k][2];
        }
        *(this_pc_buffer+i*3+0) = tmpx + 1*inverse_apex[0];
        *(this_pc_buffer+i*3+1) = tmpy + 1*inverse_apex[1];
        *(this_pc_buffer+i*3+2) = tmpz + 1*inverse_apex[2];
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
        find_triangle = get_triangle_point(point,interpolated_point, (float (*)[3])this_pc_buffer, patch_size, this_dist);
        if(!find_triangle){
            return;
        }
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

        if((length_of_cones[i]<(1-threshold)*natural_cone) || (length_of_cones[i]>(1+threshold)*natural_cone)){
            return;
        }

        // compute length of perimeters by interpolation
        // notice: keep vertice points unchanged (i*5 in perimeter)
        for(j=1;j<5;j++)
        {
            // sample a point from the perimeter
            temp1 = (float)j / 5.0;
            point[0] = (1 - temp1) * vertices[i] + temp1 * vertices[(i+1)%8];
            point[1] = (1 - temp1) * vertices[i+8] + temp1 * vertices[(i+1)%8+8];
            point[2] = (1 - temp1) * vertices[i+16] + temp1 * vertices[(i+1)%8+16];

            find_triangle = get_triangle_point(point,interpolated_point, (float (*)[3])this_pc_buffer, patch_size, this_dist);
            if(!find_triangle){
                return;
            }
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

        if((length_of_perimeters[i]<(1-threshold)*natural_perimeter) || (length_of_perimeters[i]>(1+threshold)*natural_perimeter)){
            return;
        }


        // compute length of flexions by interpolation
        // notice: keep vertice points unchanged (i*5 in flexion)
        for(j=1;j<5;j++)
        {
            // sample a point from the perimeter
            temp2 = (float)j / 5.0;
            point[0] = (1 - temp2) * vertices[i] + temp2 * vertices[(i+2)%8];
            point[1] = (1 - temp2) * vertices[i+8] + temp2 * vertices[(i+2)%8+8];
            point[2] = (1 - temp2) * vertices[i+16] + temp2 * vertices[(i+2)%8+16];

            find_triangle = get_triangle_point(point,interpolated_point, (float (*)[3])this_pc_buffer, patch_size, this_dist);
            if(!find_triangle){
                return;
            }
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

        if((length_of_flexions[i]<(1-threshold)*natural_flexion) || (length_of_flexions[i]>(1+threshold)*natural_flexion)){
            return;
        }

    }

    label_mask[id_x*width+id_y] = label_mask[id_x*width+id_y]+1;
}
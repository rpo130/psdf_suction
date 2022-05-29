from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import copy

class VacuumGripper(object):
    def __init__(self, radius=0.01, height=0.04, num_vertices=8, angle_threshold=np.pi/4):
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
        self.max_count = 10
        self.show=[]
        self.false_count=[0,0,0]
        self.perimeter_false=[]
        self.flexion_false=[]

    def get_vacuum_gripper_model(self):
        apex = np.zeros([3], dtype=np.float32)
        base_center = np.array([0.0, 0.0, self.height], dtype=np.float32)
        vertices = []
        for i in range(self.num_vertices):
            curr_angle = np.pi * 2 * i / self.num_vertices
            vertices.append([np.sin(curr_angle) * self.radius, np.cos(curr_angle) * self.radius, self.height])
        vertices = np.array(vertices, dtype=np.float32)
        return apex, base_center, vertices.T

    def is_stable(self,vision_dict, num_samples=10, threshold=0.1):
        # check if the grasp direction is upward or not
        if vision_dict['grasp_direction'][-1] >= 0:
            return False, vision_dict['grasp_center'], vision_dict['grasp_center']
        # check if the grasp direction is largely distinctive with the z direction
        if np.dot(vision_dict['grasp_direction'], np.array([0.0, 0.0, -1.0])) < np.cos(self.threshold):
            return False, vision_dict['grasp_center'], vision_dict['grasp_center']
        perimeters, flexions, cones, perimeter_points, flexion_points = self.compute_length_of_springs(grasp_point=vision_dict['grasp_center'],
                                                                                                       surface_normal=vision_dict['grasp_direction'],
                                                                                                       point_cloud=vision_dict['point_cloud'],
                                                                                                       num_samples=num_samples)

        if perimeters is None:
            return False, None, None
        # for i,perimeter in enumerate(perimeters):
        #     if perimeter > (1+threshold)*self.natural_perimeter or perimeter < (1-threshold)*self.natural_perimeter:
        #         self.false_count[0]+=1
        #         self.perimeter_false.append(i)
        #         return False, perimeter_points, flexion_points
        for i,flexion in enumerate(flexions):
            if flexion > (1+threshold) * self.natural_flexion or flexion < (1-threshold)*self.natural_flexion:
                self.false_count[1] += 1
                self.flexion_false.append(i)
                return False, perimeter_points, flexion_points
        for cone in cones:
            if cone > (1 + threshold) * self.natural_cone or cone < (1-threshold)*self.natural_cone:
                self.false_count[2] += 1
                return False, perimeter_points, flexion_points
        return True, perimeter_points, flexion_points

    def compute_length_of_springs(self, grasp_point, surface_normal, point_cloud, num_samples=10,threshold=0.1):
        """

        :param grasp_point: A 3-D numpy array representing the grasp center.
        :param surface_normal: A 3-D numpy array representing the surface normal with respect to the graps point.
        :param point_cloud: a HxWx3-D numpy array representing the point cloud in camera frame.
        :param num_samples: A int value representing the number of sampling point for each spring.
        :return: Three lists representing the projected length for each spring.
        """
        apex, _, _ = self.transform_vacuum_gripper(grasp_point, surface_normal)
        # get current transformation of the gripper
        # time cost 2~5e-4s
        transformation_g2w = self.get_transformation(apex, -surface_normal)  # gripper to world
        # time cost:2.5e-4s
        transformation_w2g = self.inverse_transformation(transformation_g2w)  # world to gripper
        # transform the point cloud from camera frame to gripper frame
        #  time cost 1e-4 s
        transformed_point_cloud = np.dot(np.concatenate([point_cloud, np.ones_like(point_cloud[..., 0:1],
                                                                                   dtype=point_cloud.dtype)],
                                                        axis=2), transformation_w2g.T)
        num_vertices = self.vertices.shape[1]
        length_of_perimeters = list()
        length_of_flexions = list()
        length_of_cones = list()
        projected_vertices = list()
        flatten_pcl = transformed_point_cloud.reshape((-1, 3))
        for i in range(num_vertices):
            # projected_vertices.append(self.get_nearest_point(self.vertices[:, i], transformed_point_cloud))
            nearest_points,find_triangle = self.get_triangle_points(self.vertices[:, i], transformed_point_cloud)
            # add by su
            if not find_triangle:
                return [None]*5

            projected_vertices.append(self.get_interpolation(nearest_points, self.vertices[:, i]))
            # plt.scatter(flatten_pcl[:, 0], flatten_pcl[:, 1], label='projected_pcl')
            # plt.scatter(self.vertices[0, :], self.vertices[1, :], label='vertices')
            # plt.scatter(nearest_points[0, :], nearest_points[1, :], label='nearest_point')
            # plt.scatter(projected_vertices[-1][0], projected_vertices[-1][1], label='prpject_point')
            # plt.show()
        perimeter_points = list()
        flexion_points = list()
        for i in range(num_vertices):
            # compute length of cone
            length_of_cones.append(np.linalg.norm(projected_vertices[i]-self.apex))
            # compute length of perimeter
            interpolation_points = list()
            interpolation_points.append(projected_vertices[i])
            for j in range(1, num_samples):
                a = j / num_samples
                # sample a point from the perimeter
                curr_point = (1 - a) * self.vertices[:, i] + a * self.vertices[:, (i + 1) % num_vertices]
                nearest_points,find_triangle = self.get_triangle_points(curr_point, transformed_point_cloud)
                # add by su
                if not find_triangle:
                    return [None]*5
                interpolation_points.append(self.get_interpolation(nearest_points, curr_point))
                # interpolation_points.append(self.get_nearest_point(curr_point, transformed_point_cloud))
            interpolation_points.append(projected_vertices[(i + 1) % num_vertices])
            perimeter_points += interpolation_points
            lengths = list(map(lambda x, y: np.linalg.norm(x - y),
                               interpolation_points[0:-1],
                               interpolation_points[1:]))
            length_of_perimeters.append(reduce(lambda x, y: x + y, lengths))
            # add by su
            if length_of_perimeters[-1] > (1 + threshold) * self.natural_perimeter or length_of_perimeters[-1] < (
                    1 - threshold) * self.natural_perimeter:
                return [None] * 5
            # compute length of flexion
            interpolation_points = list()
            interpolation_points.append(projected_vertices[i])
            for j in range(1, num_samples):
                a = j / num_samples
                # sample a point from the flexion
                curr_point = (1 - a) * self.vertices[:, i] + a * self.vertices[:, (i + 2) % num_vertices]
                nearest_points,find_triangle = self.get_triangle_points(curr_point, transformed_point_cloud)
                # add by su
                if not find_triangle:
                    return [None]*5
                interpolation_points.append(self.get_interpolation(nearest_points, curr_point))
                # interpolation_points.append(self.get_nearest_point(curr_point, transformed_point_cloud))
            interpolation_points.append(projected_vertices[(i + 2) % num_vertices])
            flexion_points += interpolation_points
            lengths = list(map(lambda x, y: np.linalg.norm(x - y),
                               interpolation_points[0:-1],
                               interpolation_points[1:]))
            length_of_flexions.append(reduce(lambda x, y: x + y, lengths))
        perimeter_points = np.dot(transformation_g2w[:, 0:-1], np.array(perimeter_points).T) + transformation_g2w[:, -1:]
        flexion_points = np.dot(transformation_g2w[:, 0:-1], np.array(flexion_points).T) + transformation_g2w[:, -1:]
        return length_of_perimeters, length_of_flexions, length_of_cones, perimeter_points, flexion_points

    def transform_vacuum_gripper(self, grasp_point, surface_normal):
        """
        Transform the vacuum gripper according to the surface normal of the given grasp point.
        We rotate the vacuum gripper frame so that the direction of z axis
        is aligned to that of the inverse direction of surface normal.
        Then we translate the vacuum gripper in which the base center is aligned to the location of grasp point.
        :param grasp_point: A 3-D numpy array representing the location of the grasp point.
        :param surface_normal: A 3-D numpy array representing the normalized surface normal of the grasp point.
        """
        inverse_surface_normal = -surface_normal
        transformation = self.get_transformation(grasp_point, inverse_surface_normal)
        # add inverse value of center base to align base center to the origin
        transformation[:, -1] += -np.dot(transformation[:, 0:-1], self.base_center)
        transformed_apex = np.dot(transformation[:, 0:-1], self.apex) + transformation[:, -1]
        transformed_base_center = np.dot(transformation[:, 0:-1], self.base_center) + transformation[:, -1]
        transformed_vertices = np.dot(transformation,
                                      np.concatenate([self.vertices, np.ones([1, self.vertices.shape[1]],
                                                                             dtype=self.vertices.dtype)],
                                                     axis=0))
        return transformed_apex, transformed_base_center, transformed_vertices

    def get_triangle_points(self, point, point_cloud):
        """
        Given the target point, the function projects the point to the point cloud surface
        and computes the topic k nearest points of point cloud for that point.
        :param point: A 3-D numpy array representing the target point in gripper space.
        :param point_cloud: A HxWx3-D numpy array representing the point cloud in gripper space.
        :return: The nearest points and their corresponding indices.
        """
        height, width, _ = np.shape(point_cloud)
        # move the point cloud so that the grasp point is on the principal axis
        horizontal_distances = point_cloud[..., 0] - point[0]
        vertical_distances = point_cloud[..., 1] - point[1]
        distances = horizontal_distances ** 2 + vertical_distances ** 2
        distances = distances.flatten()
        # theoretically 20 candidates are enough to find the triangle.
        indices = np.argsort(distances)[:20]
        indices = [[int(idx / width), idx % width] for idx in indices]
        find_triangle = False
        # Although the time complexity looks like O(N^3), the worst case is hardly happened,
        # and we can compute the triangle for only single loop at most time.
        count = 0
        #add by su:enumerate
        for ind1,idx0 in enumerate(indices):
            ind_2 = ind1+1
            for ind2,idx1 in enumerate(indices[ind_2:]):
                ind_3 = ind_2+ind2+1
                for idx2 in indices[ind_3:]:
                    count+=1
                    # move the triangle to the origin by subtracting the target point
                    find_triangle = self.is_in_triangle(point_cloud[idx0[0], idx0[1]]-point,
                                                        point_cloud[idx1[0], idx1[1]]-point,
                                                        point_cloud[idx2[0], idx2[1]]-point)
                    if find_triangle or count>100: break
                if find_triangle or count>100: break
            if find_triangle or count>100: break
        if not find_triangle:
            idx0, idx1, idx2 = indices[0], indices[1],indices[7]
        # else:
        #     if count>self.max_count:
        #         self.max_count=count
        #         print('count={}'.format(count))
        # candidate_indices = np.array([idx0, idx1, idx2])
        # candidate_points = point_cloud[candidate_indices[:, 0], candidate_indices[:, 1]].T
        # plt.scatter(candidate_points[0, :]-point[0], candidate_points[1, :]-point[1])
        # plt.scatter(0, 0)
        # plt.show()
        candidate_points = np.stack((point_cloud[idx0[0],idx0[1]],
                                    point_cloud[idx1[0],idx1[1]],
                                    point_cloud[idx2[0],idx2[1]])).T
        self.show.append(count)
        return candidate_points,find_triangle

    @staticmethod
    def get_nearest_point(point, point_cloud):
        """
        (Deprecated) Get nearest point from the point cloud for a radial line.
        Because the approximate process bring large errors, this method is deprecated.
        :param point: A 3-D numpy array representing the mass point generating the radial line.
        :param point_cloud: A HxWx3-D numpy array representing the point cloud in gripper space.
        :return: A 3-D numpy array representing the nearest point.
        """
        height, width, _ = point_cloud.shape
        # move the point cloud so that the grasp point is on the principal axis
        horizontal_distances = np.abs(point_cloud[..., 0] - point[0])
        vertical_distances = np.abs(point_cloud[..., 1] - point[1])
        distances = horizontal_distances + vertical_distances
        distances = distances.flatten()
        # theoretically 100 candidates are enough to find the triangle including the origin.
        indices = np.argsort(distances)
        index = [int(indices[0] / width), indices[0] % width]
        return point_cloud[index[0], index[1]]

    @staticmethod
    def get_transformation(point, curr_z_direction):
        """
        Get transformation according to the angle axis representation.
        The details of the formula are illustrated in chapter 7 of "State Estimation for Robotics".
        :param point: A 3-D numpy array representing the point in camera frame.
        :param curr_z_direction: the representation in camera frame for current z axis of gripper frame.
        :return: A 3x4-D numpy array representing the rotation matrix and the translation.
        """
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        # get normalized rotation axis and corresponding skew symmetric matrix
        rotation_axis = np.cross(z_axis, curr_z_direction)
        # normalize rotation axis when it is not 0 vector.
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) if np.sum(rotation_axis) != 0 else rotation_axis
        skew_symmetric_matrix = np.array([[0.0, -rotation_axis[2], rotation_axis[1]],
                                          [rotation_axis[2], 0.0, -rotation_axis[0]],
                                          [-rotation_axis[1], rotation_axis[0], 0.0]], dtype=np.float32)
        rotation_axis = np.expand_dims(rotation_axis, axis=1)
        cos = np.dot(z_axis, curr_z_direction)
        sin = np.sin(np.arccos(cos))

        # compute rotation matrix according to the axis angle representation
        rotation_matrix = cos * np.identity(3, dtype=np.float32) + \
                          (1 - cos) * np.dot(rotation_axis, rotation_axis.T) + \
                          sin * skew_symmetric_matrix
        # add inverse value of center base to align base center to the origin
        translation = np.expand_dims(point, axis=1)
        return np.concatenate([rotation_matrix, translation], axis=1)

    @staticmethod
    def is_in_triangle(point1, point2, point3):
        """
        https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
        """
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        origin = np.zeros_like(point1, dtype=point1.dtype)
        d1 = sign(origin, point1, point2)
        d2 = sign(origin, point2, point3)
        d3 = sign(origin, point3, point1)
        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return not (has_neg & has_pos)

    @staticmethod
    def get_interpolation(nearest_points, point):
        """

        :param nearest_points: A 3x3-D numpy array representing top 3 nearest points in shifted gripper space.
        :param point: A 3-D numpy array representing the target point in gripper space.
        :return: A 3-D numpy array representing the interpolation point.
        """
        interpolated_point = copy.deepcopy(point)
        # convert to homogeneous points
        nearest_points = np.concatenate([nearest_points, np.ones([1, 3], dtype=nearest_points.dtype)])  # 4x3
        # compute surface and find the interpolated point
        # Details about computing the 3d surface are illustrated in
        # Chapter 3 of Multiple View Geometry in Computer Science
        # d123 = np.linalg.det(np.stack([nearest_points[1], nearest_points[2], nearest_points[3]], axis=0))
        # d023 = np.linalg.det(np.stack([nearest_points[0], nearest_points[2], nearest_points[3]], axis=0))
        # d013 = np.linalg.det(np.stack([nearest_points[0], nearest_points[1], nearest_points[3]], axis=0))
        # d012 = np.linalg.det(np.stack([nearest_points[0], nearest_points[1], nearest_points[2]], axis=0))
        # if abs(d013)==0:
        #     a=1
        x1,y1,z1 = nearest_points[:3,0]
        x2,y2,z2 = nearest_points[:3,1]
        x3,y3,z3= nearest_points[:3,2]

        A = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        B = (x3 - x1) * (z2 - z1) - (x2 - x1) * (z3 - z1)
        C = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        D = -(A * x1 + B * y1 + C * z1)
        if C==0:
            NO = 1
            print('nearest point:')
            print(nearest_points[:3,...])
        interpolated_point[2] = -(A*point[0]+B*point[1]+D)/C
        # interpolated_point[2] = (-d123 * point[0] + d023 * point[1] + d012) / d013
        return interpolated_point

    @staticmethod
    def inverse_transformation(transformation):
        """
        Get inverse transformation from original transformation
        :param transformation: A 3x4-D numpy array representing the transformation.
        :return: A 3x4-F numpy array representing the inverse transformation.
        """
        rotation_matrix = copy.deepcopy(transformation[0:, 0:-1])
        translation = copy.deepcopy(transformation[:, 3:])
        inverse_rotation_matrix = rotation_matrix.T
        inverse_translation = -np.dot(inverse_rotation_matrix, translation)
        return np.concatenate([inverse_rotation_matrix, inverse_translation], axis=1)


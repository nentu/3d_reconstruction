import numpy as np

from src.stereo_pair.utils import get_rotation_matrix


def closest_point_to_two_lines(line1_point1, line1_point2, line2_point1, line2_point2):
    # Define the direction vectors of the lines
    line1_direction = np.array(line1_point2) - np.array(line1_point1)
    line2_direction = np.array(line2_point2) - np.array(line2_point1)

    # Normalize the direction vectors
    line1_direction = line1_direction / np.linalg.norm(line1_direction)
    line2_direction = line2_direction / np.linalg.norm(line2_direction)

    # Define the vector between the two points on the lines
    vector_between_points = np.array(line2_point1) - np.array(line1_point1)

    # Compute the cross product of the direction vectors
    cross_product = np.cross(line1_direction, line2_direction)

    # Check if the lines are parallel
    if np.linalg.norm(cross_product) < 1e-6:
        # If the lines are parallel, the closest point is the projection of one point onto the other line
        projection_onto_line1 = np.array(line1_point1) + np.dot(vector_between_points,
                                                                line1_direction) * line1_direction
        projection_onto_line2 = np.array(line2_point1) + np.dot(-vector_between_points,
                                                                line2_direction) * line2_direction
        closest_point = (projection_onto_line1 + projection_onto_line2) / 2
    else:
        # Compute the vector from the first point to the closest point
        t = np.dot(np.cross(line2_direction, vector_between_points), cross_product) / np.dot(cross_product,
                                                                                             cross_product)
        closest_point = np.array(line1_point1) + t * line1_direction

    return closest_point


def make_homogeneous(coord: np.array):
    return np.concatenate((coord, np.array([1])), axis=0)


def get_rt_matrix(rotation: np.array, translation: np.array):
    rot = get_rotation_matrix(rotation)
    i = np.eye(4, 4)
    i[:rot.shape[0], :rot.shape[1]] = rot
    i[:3, -1] = translation
    return i


def _get_camera_point_coords(camera_rt_matrix: np.array, inv_intrinsic_matrix: np.array, point):
    """
    get coordinates of camera center and point in virtual plane

    :param camera_rt_matrix: Rotation matrix .dot( translation vector ) for camera
    :param inv_intrinsic_matrix: inverse camera intrinsic matrix
    :param point: (u, v) coordinates of the point
    :return: ((x1, y1, z1), (x2, y2, z2))
    """
    camera_coord = np.array([0, 0, 0, 1])
    virtual_coord = make_homogeneous(
        inv_intrinsic_matrix.dot(
            make_homogeneous(point)
        )
    )

    camera_coord = camera_rt_matrix.dot(camera_coord)
    virtual_coord = camera_rt_matrix.dot(virtual_coord)

    return np.array([camera_coord[:3], virtual_coord[:3]])


class StereoPair(object):
    camera_rt_matrix_1: np.array
    camera_rt_matrix_2: np.array
    inv_intrinsic_matrix1: np.array
    inv_intrinsic_matrix2: np.array

    def __init__(self, camera_rt_matrix_1, camera_rt_matrix_2,
                 intrinsic_matrix1, intrinsic_matrix2):
        """

        :param camera_rt_matrix_1: Rotation matrix .dot( translation vector ) for camera1
        :param camera_rt_matrix_2: Rotation matrix .dot( translation vector ) for camera2
        :param intrinsic_matrix1:
        :param intrinsic_matrix2:
        """
        self.camera_rt_matrix_1 = camera_rt_matrix_1
        self.camera_rt_matrix_2 = camera_rt_matrix_2
        self.inv_intrinsic_matrix1 = np.linalg.inv(intrinsic_matrix1)
        self.inv_intrinsic_matrix2 = np.linalg.inv(intrinsic_matrix2)

    def compute_single_point(self, point1, point2):
        cam1, p1 = _get_camera_point_coords(self.camera_rt_matrix_1, self.inv_intrinsic_matrix1, point1)
        cam2, p2 = _get_camera_point_coords(self.camera_rt_matrix_2, self.inv_intrinsic_matrix2, point2)
        res = closest_point_to_two_lines(
            cam1, p1, cam2, p2
        )

        return res

    def compute_3d(self, point_list1, point_list2):
        """
        Compute the (x, y, z) coordinate for each pair of points point_list1[i] and point_list2[i] are the coordinates of the same 3d points

        :param point_list1: [u, v] coordinates of the first image
        :param point_list2: [u, v] coordinates of the second image
        :return: [(x, y, z)] * n coordinates of the depth
        """
        return [self.compute_single_point(p1, p2) for p1, p2 in zip(point_list1, point_list2)]


if __name__ == '__main__':
    rotation = [0, 180 + 45, 0]
    translation = [10, 0, 0]
    i = get_rt_matrix(rotation, translation)

    print(_get_camera_point_coords(i, np.eye(3, 3), [0, 0]))

    print(closest_point_to_two_lines(
        [1, 0, 0],
        [2, 0, 0],
        [1, 0, 1],
        [1, 1, 1]
    ))

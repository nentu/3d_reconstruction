import numpy as np

from src.stereo_pair.utils import get_rotation_matrix


def get_closest_point_and_dist(line1_p1, line1_p2, line2_p1, line2_p2):
    list_t = np.array(
        [
            0,
            -(line1_p1 - line1_p2).dot(line2_p1 - line2_p2),
            (line2_p2 - line2_p1).dot(line2_p1 - line2_p2),
            (line2_p2 - line1_p1).dot(line2_p1 - line2_p2),
            -(line1_p1 - line1_p2).dot(line1_p1 - line1_p2),
            (line2_p2 - line2_p1).dot(line1_p1 - line1_p2),
            (line2_p2 - line1_p1).dot(line1_p1 - line1_p2),
        ]
    )

    t1 = (list_t[2] * list_t[6] - list_t[3] * list_t[5]) / (
        list_t[1] * list_t[5] - list_t[2] * list_t[4]
    )
    t2 = (-list_t[1] * list_t[6] + list_t[3] * list_t[4]) / (
        list_t[1] * list_t[5] - list_t[2] * list_t[4]
    )

    p2_value = line2_p2 + (line2_p2 - line2_p1) * t2
    p1_value = line1_p1 + (line1_p1 - line1_p2) * t1
    return (p1_value + p2_value) / 2, np.linalg.norm(p2_value - p1_value)


def make_homogeneous(coord: np.array, axis=0):
    return np.insert(coord, coord.shape[axis], 1, axis=axis)


def get_rt_matrix(rotation: np.array, translation: np.array):
    rot = get_rotation_matrix(rotation)
    i = np.eye(4, 4)
    i[: rot.shape[0], : rot.shape[1]] = rot
    i[:3, -1] = translation
    return i


def _get_camera_point_coords(
    camera_rt_matrix: np.array, inv_intrinsic_matrix: np.array, point
):
    """
    get coordinates of camera center and point in virtual plane

    :param camera_rt_matrix: Rotation matrix .dot( translation vector ) for camera
    :param inv_intrinsic_matrix: inverse camera intrinsic matrix
    :param point: (u, v) coordinates of the point
    :return: ((x1, y1, z1), (x2, y2, z2))
    """
    camera_coord = np.array([0, 0, 0, 1])
    virtual_coord = make_homogeneous(inv_intrinsic_matrix.dot(make_homogeneous(point)))

    camera_coord = camera_rt_matrix.dot(camera_coord)
    virtual_coord = camera_rt_matrix.dot(virtual_coord)
    return np.array([camera_coord[:3], virtual_coord[:3]])


class StereoPair(object):
    camera_rt_matrix_1: np.array
    camera_rt_matrix_2: np.array
    inv_intrinsic_matrix1: np.array
    inv_intrinsic_matrix2: np.array

    def __init__(
        self,
        camera_rt_matrix_1,
        camera_rt_matrix_2,
        intrinsic_matrix1,
        intrinsic_matrix2,
    ):
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
        cam1, p1 = _get_camera_point_coords(
            self.camera_rt_matrix_1, self.inv_intrinsic_matrix1, point1
        )
        cam2, p2 = _get_camera_point_coords(
            self.camera_rt_matrix_2, self.inv_intrinsic_matrix2, point2
        )
        res = get_closest_point_and_dist(cam1, p1, cam2, p2)[0]

        return res

    def compute_3d(self, point_list1, point_list2):
        """
        Compute the (x, y, z) coordinate for each pair of points point_list1[i] and point_list2[i] are the coordinates of the same 3d points

        :param point_list1: [u, v] coordinates of the first image
        :param point_list2: [u, v] coordinates of the second image
        :return: [(x, y, z)] * n coordinates of the depth
        """
        return np.array(
            [
                self.compute_single_point(p1, p2)
                for p1, p2 in zip(point_list1, point_list2)
            ]
        )


if __name__ == "__main__":
    rotation = [0, 180 + 45, 0]
    translation = [10, 0, 0]
    i = get_rt_matrix(rotation, translation)

    print(_get_camera_point_coords(i, np.eye(3, 3), [0, 0]))

    print(get_closest_point_and_dist([1, 0, 0], [2, 0, 0], [1, 0, 1], [1, 1, 1]))

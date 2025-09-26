import numpy as np
from scipy.spatial.transform import Rotation


def get_intrinsic_matrix(angle_x, angle_y, w, h):
    fx = w / np.tan(angle_x / 2)
    fy = h / np.tan(angle_y / 2)

    cx = w / 2
    cy = h / 2

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def project_point(intrinsic_matrix, point):
    uv = intrinsic_matrix.dot(point)
    return uv[:2] / uv[2]


def rotate(model, x, y, z, invert=False):
    matrix = Rotation.from_euler("zyx", [z, y, x], degrees=True).as_matrix()
    if invert:
        matrix = np.linalg.inv(matrix)
    model.vertex_list = model.vertex_list.dot(matrix)

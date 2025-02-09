import cv2
import numpy as np

from src.basic_graphic.cube import Cube
from src.basic_graphic.draw_utils import draw_model, get_depth_map
from src.basic_graphic.utils import get_intrinsic_matrix, rotate
from src.basic_graphic.model import Model
from src.stereo_pair.stereo_pair import StereoPair, get_rt_matrix, make_homogeneous


def _draw_model(plane_shape, model, name):
    model.apply_tranform(intrinsic_matrix)
    plane = get_depth_map(plane_shape, model)

    draw_model(plane, model)
    cv2.imshow(name, plane)


if __name__ == "__main__":
    plane_shape = np.array([300, 300])
    camera_angle = (90, 90)

    left_camera_pos = np.array([-30, -10, 10])
    right_camera_pos = np.array([22, 14, -14])

    left_camera_rot = np.array([0, 0, 0])
    right_camera_rot = np.array([0, 0, 0])

    intrinsic_matrix = get_intrinsic_matrix(*camera_angle, *plane_shape)

    left_rt_matrix = get_rt_matrix(left_camera_rot, left_camera_pos)
    right_rt_matrix = get_rt_matrix(right_camera_rot, right_camera_pos)

    inv_left_rt_matrix = np.linalg.inv(left_rt_matrix)
    inv_right_rt_matrix = np.linalg.inv(right_rt_matrix)
    sp = StereoPair(
        left_rt_matrix,
        right_rt_matrix,
        intrinsic_matrix,
        intrinsic_matrix,
    )

    r = 100

    # print(angle_coordinates)

    shift = 0
    rotation = 180

    h = 0
    k = 1

    minus_eye = -1 * np.eye(4, 4)
    minus_eye[-1][-1] = 1

    err_rt_matrix = get_rt_matrix(
        right_camera_rot + left_camera_rot, right_camera_pos + left_camera_pos
    ).dot(minus_eye)

    while True:
        model = Cube(r) # ObjModel('../graphic/models/cow.obj')

        general_rt_matrix = get_rt_matrix([180, rotation, 0], [0, 0, 300 + shift])
        model.vertex_list = general_rt_matrix.dot(
            make_homogeneous(model.vertex_list, 1).T
        ).T[:, :3]

        # rotate(model, 180, rotation, 0)

        # model.vertex_list[:, 2] += 300 + shift

        model_r = model.copy()
        model_l = model.copy()

        model_l.vertex_list = inv_left_rt_matrix.dot(
            make_homogeneous(model_l.vertex_list, 1).T
        ).T[:, :3]

        model_r.vertex_list = inv_right_rt_matrix.dot(
            make_homogeneous(model_r.vertex_list, 1).T
        ).T[:, :3]

        # model_r.vertex_list -= inv_right_rt_matrix.dot(make_homogeneous(model_r.vertex_list, 1))

        _draw_model(plane_shape, model_l, "camera_left")
        _draw_model(plane_shape, model_r, "camera_right")

        res = sp.compute_3d(
            model_r.vertex_list[:, :2],  #  .astype(np.int16),
            model_l.vertex_list[:, :2],  #  .astype(np.int16),
        )
        # res *= -1
        # res += left_camera_pos + right_camera_pos
        res = err_rt_matrix.dot(make_homogeneous(res, 1).T).T[:, :3]

        res_model = Cube(0)
        res_model.vertex_list = np.array(res)

        er = np.abs(model.vertex_list - res)
        if np.linalg.norm(er) > 1e-2:
            print(res[0])
            print(np.mean(er, axis=0))
            print(np.std(er, axis=0))
        # print("==")
        _draw_model(plane_shape, res_model, "res")

        # _draw_model(plane_shape, model, "camera_center", True)

        if cv2.waitKey(0) == ord("q"):
            break

        shift += 0
        rotation += 0.1
        # k += 2e-3

    cv2.destroyAllWindows()

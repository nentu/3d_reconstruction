import cv2
import numpy as np

from src.basic_graphic.models.cube import Cube
from src.basic_graphic.models.camera import Camera
from src.basic_graphic.draw_utils import draw_model, get_depth_map
from src.basic_graphic.utils import get_intrinsic_matrix
from src.stereo_pair.stereo_pair import StereoPair, get_rt_matrix, make_homogeneous


def _draw_model(plane_shape, model, name, intrinsic_matrix):
    projected_matrix = model.apply_tranform(intrinsic_matrix)
    plane = get_depth_map(plane_shape, projected_matrix)
    plane += 255
    draw_model(plane, projected_matrix)
    cv2.imshow(name, plane)
    return projected_matrix


def get_rt_matrix_pair(pos: np.array, rot: np.array):
    rt_matrix = get_rt_matrix(rot, pos)
    inv_rt_matrix = np.linalg.inv(rt_matrix)
    return rt_matrix, inv_rt_matrix


def apply_matrix_to_model(model, matrix):
    return matrix.dot(make_homogeneous(model.vertex_list, 1).T).T[:, :3]


if __name__ == "__main__":
    win_size = 400

    main_plane_shape = np.array([win_size, win_size])
    camera_plane_shape = main_plane_shape // 2
    camera_angle = (90, 90)

    main_intrinsic_matrix = get_intrinsic_matrix(*camera_angle, *main_plane_shape)
    camera_intrinsic_matrix = get_intrinsic_matrix(*camera_angle, *camera_plane_shape)

    main_rt_matrix, inv_main_rt_matrix = get_rt_matrix_pair(
        rot=np.array([0, 0, 0]), pos=np.array([0, 0, 0])
    )

    left_rt_matrix, inv_left_rt_matrix = get_rt_matrix_pair(
        rot=np.array([20, 0, 0]), pos=np.array([-31, 40, 10])
    )

    right_rt_matrix, inv_right_rt_matrix = get_rt_matrix_pair(
        rot=np.array([0, 0, 0]), pos=np.array([10, 24, -354])
    )

    # Class instance for 3d coordinates computing
    sp = StereoPair(
        left_rt_matrix,
        right_rt_matrix,
        camera_intrinsic_matrix,
        camera_intrinsic_matrix,
    )

    r = 99  # Cube size

    rotation = 180

    while True:
        # Create cameras models
        cam1_model = Camera(r / 8)
        cam2_model = Camera(r / 8)
        # Create cube model
        model = Cube(r)  # ObjModel('../graphic/models/cow.obj')

        # Move cameras' models
        cam1_model.vertex_list = apply_matrix_to_model(cam1_model, left_rt_matrix)
        cam2_model.vertex_list = apply_matrix_to_model(cam2_model, right_rt_matrix)

        # Move cube
        cube_rt_matrix = get_rt_matrix([180, rotation, 0], [0, 0, 300])
        model.vertex_list = apply_matrix_to_model(model, cube_rt_matrix)

        # Create cube models copy for each camera
        model_r = model.copy()
        model_l = model.copy()

        # Move cameras' views
        model_l.vertex_list = apply_matrix_to_model(model_l, inv_left_rt_matrix)
        model_r.vertex_list = apply_matrix_to_model(model_r, inv_right_rt_matrix)

        # Get 3d coordinates
        res = sp.compute_3d(
            model_l.apply_tranform(camera_intrinsic_matrix).vertex_list[
                :, :2
            ],  #  .astype(np.int16),
            model_r.apply_tranform(camera_intrinsic_matrix).vertex_list[
                :, :2
            ],  #  .astype(np.int16),
        )
        res_model = Cube(r)
        res_model.vertex_list = np.array(res)

        # Compute error
        er = np.abs(model.vertex_list - res)
        if np.linalg.norm(er) > 1e-2:
            print("Failed")
            print(np.mean(er, axis=0).astype(np.int16))
            print(np.std(er, axis=0))

        # Draw models
        _draw_model(main_plane_shape, res_model, "res", main_intrinsic_matrix)
        _draw_model(camera_plane_shape, model_l, "camera_left", camera_intrinsic_matrix)
        _draw_model(
            camera_plane_shape, model_r, "camera_right", camera_intrinsic_matrix
        )

        if cv2.waitKey(1) == ord("q"):
            break

        rotation += 0.1

    cv2.destroyAllWindows()

import cv2
import numpy as np

from src.basic_graphic.models.cube import Cube
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


if __name__ == "__main__":
    win_size = 400
    main_plane_shape = np.array([win_size, win_size])
    camera_plane_shape = main_plane_shape // 2

    camera_angle = (90, 90)

    left_camera_pos = np.array([-31, 40, 10])
    right_camera_pos = np.array([10, 24, -354])

    left_camera_rot = np.array([20, 0, 0])
    right_camera_rot = np.array([0, 0, 0])

    main_intrinsic_matrix = get_intrinsic_matrix(*camera_angle, *main_plane_shape)
    camera_intrinsic_matrix = get_intrinsic_matrix(*camera_angle, *camera_plane_shape)

    left_rt_matrix = get_rt_matrix(left_camera_rot, left_camera_pos)
    right_rt_matrix = get_rt_matrix(right_camera_rot, right_camera_pos)

    inv_left_rt_matrix = np.linalg.inv(left_rt_matrix)
    inv_right_rt_matrix = np.linalg.inv(right_rt_matrix)

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
        model = Cube(r)  # ObjModel('../graphic/models/cow.obj')

        # Move cube
        cube_rt_matrix = get_rt_matrix([180, rotation, 0], [0, 0, 300])
        model.vertex_list = cube_rt_matrix.dot(
            make_homogeneous(model.vertex_list, 1).T
        ).T[:, :3]

        model_r = model.copy()
        model_l = model.copy()

        # Move cameras
        model_l.vertex_list = inv_left_rt_matrix.dot(
            make_homogeneous(model_l.vertex_list, 1).T
        ).T[:, :3]

        model_r.vertex_list = inv_right_rt_matrix.dot(
            make_homogeneous(model_r.vertex_list, 1).T
        ).T[:, :3]

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
        _draw_model(camera_plane_shape, model_r, "camera_right", camera_intrinsic_matrix)

        if cv2.waitKey(1) == ord("q"):
            break

        rotation += 0.1

    cv2.destroyAllWindows()

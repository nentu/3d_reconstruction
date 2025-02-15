import cv2
import numpy as np

from src.basic_graphic.models.cube import Cube
from src.basic_graphic.models.camera import Camera
from src.basic_graphic.draw_utils import draw_model, get_depth_map, color_list
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
    win_size = 800

    main_plane_shape = np.array([win_size, win_size])
    camera_plane_shape = main_plane_shape // 2
    camera_angle = (90, 90)

    main_intrinsic_matrix = get_intrinsic_matrix(*camera_angle, *main_plane_shape)
    camera_intrinsic_matrix = get_intrinsic_matrix(*camera_angle, *camera_plane_shape)

    main_rt_matrix, inv_main_rt_matrix = get_rt_matrix_pair(
        rot=np.array([-20, 25, 0]), pos=np.array([-130, -130, -230]) / 2
    )

    left_rt_matrix, inv_left_rt_matrix = get_rt_matrix_pair(
        rot=np.array([10, 10, 45]), pos=np.array([-31, 50, 0])
    )

    right_rt_matrix, inv_right_rt_matrix = get_rt_matrix_pair(
        rot=np.array([0, -20, 0]), pos=np.array([100, 24, -45])
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
    pose = 0
    while True:  # rotation < 180 + 90:
        # Create cameras models
        cam1_model = Camera(r / 4)
        cam2_model = Camera(r / 4)
        # Create cube model
        obj_model = Cube(r)  # ObjModel('../graphic/models/cow.obj')

        # Move cameras' models
        cam1_model.vertex_list = apply_matrix_to_model(cam1_model, left_rt_matrix)
        cam2_model.vertex_list = apply_matrix_to_model(cam2_model, right_rt_matrix)

        # Move cube
        cube_rt_matrix = get_rt_matrix([180, rotation, 0], [0, 0, 300])
        obj_model.vertex_list = apply_matrix_to_model(obj_model, cube_rt_matrix)

        # Create cube models copy for each camera
        model_r = obj_model.copy()
        model_l = obj_model.copy()

        # Move cameras' views
        model_l.vertex_list = apply_matrix_to_model(model_l, inv_left_rt_matrix)
        model_r.vertex_list = apply_matrix_to_model(model_r, inv_right_rt_matrix)

        # Get 3d coordinates
        res = sp.compute_3d(
            model_l.apply_tranform(camera_intrinsic_matrix)
            .vertex_list[:, :2]
            .astype(np.int16),
            model_r.apply_tranform(camera_intrinsic_matrix)
            .vertex_list[:, :2]
            .astype(np.int16),
        )
        res_model = Cube(r)
        res_model.vertex_list = np.array(res)

        # Compute error
        er = np.abs(obj_model.vertex_list - res)
        if np.linalg.norm(er) > 1e1:
            print("Failed")
            print(np.mean(er, axis=0).astype(np.int16))
            print(np.std(er, axis=0))

        # Move models for main view
        scene = (
            np.zeros(
                shape=(*(main_plane_shape * [1, 1.5]).astype(np.int32), 3),
                dtype=np.uint8,
            )
            + 255
        )

        # Draw full scene
        for model, color_i in zip(
            [obj_model, res_model, cam1_model, cam2_model], range(4)
        ):
            model = model.copy()
            model.vertex_list = apply_matrix_to_model(model, inv_main_rt_matrix)
            projected_matrix = model.apply_tranform(main_intrinsic_matrix)
            if color_i == 0:
                draw_model(scene, projected_matrix, color_i + 1, 10)
            else:
                draw_model(scene, projected_matrix, color_i + 1)

        # Draw views from camera
        p_cam1_model = model_l.apply_tranform(camera_intrinsic_matrix)
        p_cam2_model = model_r.apply_tranform(camera_intrinsic_matrix)

        draw_model(
            scene[: main_plane_shape[1] // 2, main_plane_shape[0] :], p_cam1_model, 1
        )
        draw_model(
            scene[main_plane_shape[1] // 2 :, main_plane_shape[0] :], p_cam2_model, 1
        )

        cv2.rectangle(
            scene,
            (0, 0),
            (main_plane_shape[0] - 2, scene.shape[0] - 2),
            color_list[1],
            2,
        )

        cv2.rectangle(
            scene,
            (main_plane_shape[0], 0),
            (scene.shape[1], main_plane_shape[1] // 2 - 1),
            color_list[3],
            3,
        )

        cv2.rectangle(
            scene,
            (main_plane_shape[0], main_plane_shape[1] // 2 + 1),
            (scene.shape[1] - 1, main_plane_shape[1] - 1),
            color_list[4],
            3,
        )

        cv2.imshow("Main", scene)

        if cv2.waitKey(1) == ord("q"):
            break

        rotation += 0.1
        pose += 0

    cv2.destroyAllWindows()

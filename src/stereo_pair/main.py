import cv2
import numpy as np

from src.basic_graphic.cube import Cube
from src.basic_graphic.draw_utils import draw_model, get_depth_map
from src.basic_graphic.utils import get_intrinsic_matrix, rotate

from src.stereo_pair.stereo_pair import StereoPair, get_rt_matrix


def _draw_model(plane_shape, model, name, depth = False):
    model.apply_tranform(intrinsic_matrix)

    plane = np.zeros(shape=(*plane_shape, 3))
    draw_model(plane, model)
    cv2.imshow(name, plane)
    if depth:
        depth = get_depth_map(plane_shape, model)
        cv2.imshow(name+"_depth", depth)


if __name__ == "__main__":
    plane_shape = np.array([300, 300])
    camera_angle = (90, 90)

    pair_dist = 33
    intrinsic_matrix = get_intrinsic_matrix(*camera_angle, *plane_shape)

    sp = StereoPair(
        get_rt_matrix([0, 0, 0], [- pair_dist / 2, 0, 0]),
        get_rt_matrix([0, 0, 0], [pair_dist / 2, 0, 0]),
        intrinsic_matrix,
        intrinsic_matrix
    )

    r = 100

    # print(angle_coordinates)

    shift = 0
    rotation = 180

    h = 0
    k = 1

    while True:
        model = Cube(1)  # ObjModel('../graphic/models/cow.obj')

        rotate(model, 180, rotation, 0)

        model.vertex_list *= r

        model.vertex_list[:, 2] += 300 + shift

        model_r = model.copy()
        model_l = model.copy()

        model_l.vertex_list[:, 0] += pair_dist / 2
        model_r.vertex_list[:, 0] -= pair_dist / 2

        _draw_model(plane_shape, model_l, "camera_left")
        _draw_model(plane_shape, model_r, "camera_right")

        res = sp.compute_3d(model_r.vertex_list[:, :2], model_l.vertex_list[:, :2])

        res_model = Cube(0)
        res_model.vertex_list = np.array(res)

        _draw_model(plane_shape, res_model, "res", False)

        # _draw_model(plane_shape, model, "camera_center", True)



        if cv2.waitKey(1) == ord("q"):
            break

        shift += 0
        rotation += 0.02
        # k += 2e-3

    cv2.destroyAllWindows()

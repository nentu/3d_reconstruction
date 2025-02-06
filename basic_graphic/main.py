import cv2
import numpy as np

from draw_utils import draw_model, get_depth_map
from obj_read import ObjModel
from utils import get_intrinsic_matrix, rotate
from cube import Cube


if __name__ == "__main__":
    plane_shape = np.array([640, 640])
    camera_angle = (90, 90)

    intrinsic_matrix = get_intrinsic_matrix(*camera_angle, *plane_shape)
    # print(angle_coordinates)

    shift = 0
    rotation = 180

    h = 0
    k = 1

    while True:
        model = Cube(99)

        # model.vertex_list[:, 1] -= 0.5


        rotate(model, 180, rotation, 0)

        model.vertex_list[:, 2] += 300 + shift
        plane = np.zeros(shape=(*plane_shape, 3))

        # plane += 255

        model.apply_tranform(intrinsic_matrix)
        depth = get_depth_map(plane_shape, model)
        depth = cv2.cvtColor(depth, cv2.COLOR_GRAY2RGB)
        draw_model(depth, model)
        draw_model(plane, model)

        cv2.imshow("res", plane)
        cv2.imshow("depth", depth)
        if cv2.waitKey(1) == ord("q"):
            break

        shift += 0
        rotation += 0.2
        # k += 2e-3

    cv2.destroyAllWindows()

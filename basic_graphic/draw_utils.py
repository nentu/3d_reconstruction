import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from model import Model


def draw_point(img, coord: np.array):
    cv2.circle(img, coord, 1, (0, 255, 0), -1)


def draw_line(img, p1, p2):
    cv2.line(img, p1, p2, (0, 0, 255), 1, 1, 0)


def draw_poly(img, points):
    cv2.fillPoly(img, [points], (255, 0, 0), 1, 1, 1)


def draw_model(plane, model: Model):
    # for p in model.vertex_list:
    #     assert abs(p[2]) == 1, f"point: {p}"

    vertex_list = model.vertex_list.astype(np.int32)
    vertex_list = vertex_list[vertex_list[:, 2] > 0]
    #
    # for p1, p2, p3 in model.poly_list_id:
    #     draw_line(plane, *vertex_list[[p1, p2]][:, :2])
    #     draw_line(plane, *vertex_list[[p1, p3]][:, :2])
    #     draw_line(plane, *vertex_list[[p2, p3]][:, :2])

    for edges in model.edges_list_id:
        draw_line(plane, *vertex_list[edges][:, :2])

    for i in vertex_list:
        draw_point(plane, i[:2])


def get_depth_map(plane_shape, model):
    known_points = model.vertex_list
    # Separate the coordinates and depths
    coords = known_points[:, :2]
    depths = known_points[:, 2]  # np.sqrt(np.sum(np.power(known_points, 2), axis=1))

    # Create a linear interpolator
    interpolator = LinearNDInterpolator(coords, depths)

    # Define the grid of points to interpolate
    x = np.arange(0, plane_shape[1])  # Assuming 256x256 image
    y = np.arange(0, plane_shape[0])
    grid_x, grid_y = np.meshgrid(x, y)

    # Interpolate the depths
    interpolated_depths = interpolator(grid_x, grid_y)

    interpolated_depths = interpolated_depths / 500 * 250

    interpolated_depths = np.nan_to_num(interpolated_depths, nan=255)

    # Replace NaN values (outside the convex hull) with 255 (background)
    interpolated_depths = np.nan_to_num(interpolated_depths, nan=255)

    # Ensure depths are within the valid range [0, 255]
    interpolated_depths = np.clip(interpolated_depths, 0, 255)

    # Convert to uint8 to match the original depth values
    interpolated_depths = interpolated_depths.astype(np.uint8)

    return interpolated_depths

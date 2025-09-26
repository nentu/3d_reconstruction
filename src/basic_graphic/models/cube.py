from models.model import Model
import numpy as np


def _gen_edge_seq():
    res = list()
    for i in range(4):  # upper plane
        res.append([i + 4, (i + 1) % 4 + 4])
    for i in range(4):  # down plane
        res.append([i, i + 4])
    for i in range(4):  # vertical lines
        res.append([i, (i + 1) % 4])
    # for i in range(8):  # vertical lines
    #     res.append([i, 8])

    return np.array(res)


def _gen_poly_seq():
    return np.array(
        [
            # Back face
            [0, 1, 2],
            [0, 2, 3],
            # Front face
            [4, 5, 6],
            [4, 6, 7],
            # Left face
            [0, 4, 7],
            [0, 7, 3],
            # Right face
            [1, 5, 6],
            [1, 6, 2],
            # Top face
            [3, 7, 6],
            [3, 6, 2],
            # Bottom face
            [0, 4, 5],
            [0, 5, 1],
        ]
    )


def _get_cube(r):
    return np.array(
        [
            [r, r, r],
            [-r, r, r],
            [-r, -r, r],
            [r, -r, r],
            [r, r, -r],
            [-r, r, -r],
            [-r, -r, -r],
            [r, -r, -r],
        ]
    ).astype(np.float32)


class Cube(Model):
    def __init__(self, r):
        vertex_list = _get_cube(r)
        super().__init__(vertex_list, _gen_edge_seq(), _gen_poly_seq())


if __name__ == "__main__":
    print(Cube(1).edges_list_id)

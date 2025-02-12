from src.basic_graphic.models.model import Model
import numpy as np


def _gen_edge_seq():
    res = list()
    for i in range(4):  # vertical lines
        res.append([1 + i, 1 + ((i + 1) % 4)])

    for i in range(4):  # vertical lines
        res.append([0, i + 1])

    return np.array(res)


def _get_vertex(r):
    res = [[0, 0, 0]]
    for x in range(2):
        for y in range(2):
            x1 = r / 2 * (2 * x - 1)
            y1 = r / 2 * (2 * y - 1)
            res.append([x1, y1, r])
    return np.array(res)


class Camera(Model):
    def __init__(self, r):
        vertex_list = _get_vertex(r)
        super().__init__(vertex_list, _gen_edge_seq(), list())


if __name__ == "__main__":
    print(Camera(1).edges_list_id)

import numpy as np
from dataclasses import dataclass


@dataclass
class Model:
    vertex_list: np.array
    edges_list_id: np.array
    poly_list_id: np.array

    def apply_tranform(self, matrix: np.array):
        self.vertex_list = matrix.dot(self.vertex_list.T).T
        self.vertex_list[:, :2] = self.vertex_list[:, :2] / np.expand_dims(
            self.vertex_list[:, 2], 1
        )

    def copy(self):
        return Model(self.vertex_list.copy(), self.edges_list_id, self.poly_list_id)


if __name__ == "__main__":
    print(Model([1, 2], [], []).vertex_list)

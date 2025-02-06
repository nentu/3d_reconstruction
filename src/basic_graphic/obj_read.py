from model import Model
import re
import numpy as np


class ObjModel(Model):
    def __init__(self, filepath):
        f = open(filepath, "r").read()
        vertex = np.array(
            [
                list(map(float, i))
                for i in re.findall("v (-?\d+\.\d+) (-?\d+\.\d+) (-?\d+\.\d+)", f)
            ]
            + [[0, 0, 0]]
        )

        edges = (
            np.array([list(map(int, i)) for i in re.findall("(\d+)/\d+ (\d+)/\d+", f)])
            - 1
        )

        s = np.max(np.abs(vertex))
        super().__init__(vertex / s, edges, [])

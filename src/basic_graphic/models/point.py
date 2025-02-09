from src.basic_graphic.models.model import Model
import numpy as np


class Point(Model):
    def __init__(self, x=0, y=0, z=0):
        vertex_list = np.array([[x, y, z]])
        super().__init__(vertex_list, np.array([]), np.array([]))

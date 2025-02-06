from scipy.spatial.transform import Rotation

def get_dict_from_r_params(r_params):
    return {s: r for s, r in zip("xyz", r_params)}


def reorder_r_params(r_params: list, order: str):
    d = get_dict_from_r_params(r_params)
    return [d[order[i]] for i in range(3)]


def get_rotation_matrix(r_params, rotation_order="zyx"):
    """
    r_params: rotation params: [x, y, z]
    """

    r = Rotation.from_euler(
        rotation_order, reorder_r_params(r_params, rotation_order), degrees=True
    )
    rotation_matrix = r.as_matrix()
    # rotation_matrix[2] = [0, 0, 1]

    return rotation_matrix
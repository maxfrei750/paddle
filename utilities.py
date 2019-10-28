import numpy as np


def extract_bounding_box(mask):
    pos = np.where(mask)

    x_min = np.min(pos[1])
    x_max = np.max(pos[1])
    y_min = np.min(pos[0])
    y_max = np.max(pos[0])

    return [x_min, y_min, x_max, y_max]

from itertools import compress

import numpy as np
from skimage.segmentation import clear_border


def filter_border_particles(masks, scores=None):
    cleared_masks = [clear_border(mask, buffer_size=2) for mask in masks]
    do_keep = [np.any(mask) for mask in cleared_masks]
    masks = list(compress(masks, do_keep))

    if scores:
        scores = list(compress(scores, do_keep))
        return masks, scores

    return masks


def calculate_area_equivalent_diameters(masks):
    masks = np.array(masks)
    masks = masks.reshape(masks.shape[0], -1)
    areas = np.asarray(masks).sum(axis=1)
    return list(np.sqrt(4 * areas / np.pi))

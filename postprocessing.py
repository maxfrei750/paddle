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

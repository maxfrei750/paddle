from itertools import compress

import numpy as np
from skimage.segmentation import clear_border


def filter_border_particles(annotation):
    masks = annotation["masks"]

    cleared_masks = [clear_border(mask, buffer_size=2) for mask in masks]
    do_keep = [np.any(mask) for mask in cleared_masks]

    annotation = filter_annotation(annotation, do_keep)

    return annotation


def filter_annotation(annotation, do_keep):
    for key, value in annotation.items():
        if key in ["scores", "masks", "boxes", "labels"]:
            annotation[key] = list(compress(annotation[key], do_keep))

    return annotation


def calculate_area_equivalent_diameters(masks):
    masks = np.array(masks)
    masks = masks.reshape(masks.shape[0], -1)
    areas = np.asarray(masks).sum(axis=1)
    return list(np.sqrt(4 * areas / np.pi))

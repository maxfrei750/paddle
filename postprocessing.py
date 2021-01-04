from itertools import compress

import numpy as np
from diplib.PyDIP_bin import MeasurementTool as PyDipMeasurementTool
from skimage.segmentation import clear_border


def filter_border_particles(annotation):
    masks = annotation["masks"]

    cleared_masks = [clear_border(mask, buffer_size=2) for mask in masks]
    do_keep = [np.any(mask) for mask in cleared_masks]

    annotation = filter_annotation(annotation, do_keep)

    return annotation


def filter_low_score_particles(annotation, score_threshold):
    scores = annotation["scores"]
    do_keep = [score >= score_threshold for score in scores]

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


def calculate_minimum_and_maximum_feret_diameter(mask):
    mask = mask.astype(bool).astype("uint16")
    measurement = PyDipMeasurementTool.Measure(mask, mask, ["Feret"])
    feret_diameter_max = measurement[1]["Feret"][0]
    feret_diameter_min = measurement[1]["Feret"][1]

    return feret_diameter_min, feret_diameter_max


def calculate_minimum_feret_diameter(mask):
    return calculate_minimum_and_maximum_feret_diameter(mask)[0]


def calculate_maximum_feret_diameter(mask):
    return calculate_minimum_and_maximum_feret_diameter(mask)[1]


def calculate_minimum_feret_diameters(masks):
    return [calculate_minimum_feret_diameter(mask) for mask in masks]


def calculate_maximum_feret_diameters(masks):
    return [calculate_maximum_feret_diameter(mask) for mask in masks]

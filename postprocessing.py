from itertools import compress
from typing import List, Tuple

import numpy as np
from diplib.PyDIP_bin import MeasurementTool as PyDipMeasurementTool
from skimage.segmentation import clear_border

from custom_types import Annotation, Mask


def filter_border_instances(annotation: Annotation, border_width: int = 2) -> Annotation:
    """Remove instances that touch the image border.

    :param annotation: Annotation with masks.
    :param border_width: The width of the border examined.
    :return: Annotation, where border instances have been removed.
    """
    masks = annotation["masks"]

    cleared_masks = [clear_border(mask, buffer_size=border_width) for mask in masks]
    do_keep = [np.any(mask) for mask in cleared_masks]

    annotation = filter_annotation(annotation, do_keep)

    return annotation


def filter_low_score_instances(annotation: Annotation, score_threshold: float) -> Annotation:
    """Remove all instances with a score below `score_threshold`.

    :param annotation: Annotation with scores.
    :param score_threshold: Threshold, below which instances are removed.
    :return: Annotation, where instances with a score below `score_threshold` have been removed.
    """
    scores = annotation["scores"]
    do_keep = [score >= score_threshold for score in scores]

    annotation = filter_annotation(annotation, do_keep)

    return annotation


def filter_annotation(annotation: Annotation, do_keep: List[bool]) -> Annotation:
    """Filter instances based on a boolean list.

    :param annotation: Annotation
    :param do_keep:
    :return:
    """
    for key, value in annotation.items():
        if key in ["scores", "masks", "boxes", "labels"]:
            # TODO: Fix type warning.
            annotation[key] = list(compress(annotation[key], do_keep))

    return annotation


def calculate_area_equivalent_diameters(masks: List[Mask]) -> List[float]:
    """Calculate area equivalent diameters for a list of masks.

    :param masks: List of HxW numpy arrays, each of which stores an instance mask.
    :return: List of area equivalent diameters.
    """
    masks = np.array(masks)
    masks = masks.reshape(masks.shape[0], -1)
    areas = np.asarray(masks).sum(axis=1)
    return list(np.sqrt(4 * areas / np.pi))


def calculate_minimum_and_maximum_feret_diameter(mask: Mask) -> Tuple[float, float]:
    """Calculate the maximum and minimum Feret diameter of a mask.

    :param mask: HxW numpy array, which stores an instance mask.
    :return: Minimum and maximum Feret diameter of the mask.
    """
    mask = mask.astype(bool).astype("uint16")
    measurement = PyDipMeasurementTool.Measure(mask, mask, ["Feret"])
    feret_diameter_max = measurement[1]["Feret"][0]
    feret_diameter_min = measurement[1]["Feret"][1]

    return feret_diameter_min, feret_diameter_max


def calculate_minimum_feret_diameter(mask: Mask) -> float:
    """Calculate the minimum Feret diameter of a mask.

    :param mask: HxW numpy array, which stores an instance mask.
    :return: Minimum Feret diameter of the mask.
    """
    return calculate_minimum_and_maximum_feret_diameter(mask)[0]


def calculate_maximum_feret_diameter(mask: Mask) -> float:
    """Calculate the maximum Feret diameter of a mask.

    :param mask: HxW numpy array, which stores an instance mask.
    :return: Maximum Feret diameter of the mask.
    """
    return calculate_minimum_and_maximum_feret_diameter(mask)[1]


def calculate_minimum_feret_diameters(masks: List[Mask]) -> List[float]:
    """Calculates the minimum Feret diameters of a list of masks.

    :param masks: List of HxW numpy arrays, each of which stores an instance mask.
    :return: List of minimum Feret diameter of the masks.
    """
    return [calculate_minimum_feret_diameter(mask) for mask in masks]


def calculate_maximum_feret_diameters(masks: List[Mask]) -> List[float]:
    """Calculate the maximum Feret diameter of a mask.

    :param masks: List of HxW numpy arrays, each of which stores an instance mask.
    :return: List of maximum Feret diameters of the masks.
    """
    return [calculate_maximum_feret_diameter(mask) for mask in masks]

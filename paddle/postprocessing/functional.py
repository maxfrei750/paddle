from typing import List, Tuple, Union

import numpy as np
from diplib.PyDIP_bin import MeasurementTool as PyDipMeasurementTool
from numpy import ndarray
from skimage.segmentation import clear_border
from torch import Tensor

from ..custom_types import Annotation, Mask


def filter_border_instances(annotation: Annotation, border_width: int = 2) -> Annotation:
    """Remove instances that touch the image border.

    :param annotation: Annotation with masks.
    :param border_width: The width of the border examined.
    :return: Annotation, where border instances have been removed.
    """
    masks = annotation["masks"]

    masks = np.asarray(masks)

    cleared_masks = np.asarray(
        [clear_border(mask, buffer_size=border_width) for mask in masks.astype(bool)]
    )

    do_keep = ~np.any(np.logical_xor(cleared_masks, masks), axis=(1, 2))

    annotation = filter_annotation(annotation, do_keep)

    return annotation


def filter_empty_instances(annotation: Annotation) -> Annotation:
    """Remove instances with empty masks.

    :param annotation: Annotation with masks.
    :return: Annotation, where instances with empty masks have been removed.
    """
    masks = annotation["masks"]

    masks = np.asarray(masks)

    do_keep = np.any(masks.squeeze(), axis=(1, 2))

    annotation = filter_annotation(annotation, do_keep)

    return annotation


def filter_low_score_instances(annotation: Annotation, score_threshold: float) -> Annotation:
    """Remove all instances with a score below `score_threshold`.

    :param annotation: Annotation with scores.
    :param score_threshold: Threshold, below which instances are removed.
    :return: Annotation, where instances with a score below `score_threshold` have been removed.
    """
    scores = annotation["scores"]

    if not (isinstance(scores, ndarray) or isinstance(scores, Tensor)):
        scores = np.asarray(scores)

    do_keep = scores >= score_threshold

    annotation = filter_annotation(annotation, do_keep)

    return annotation


def filter_class_instances(annotation: Annotation, class_labels_to_keep: List[int]) -> Annotation:
    """Remove all instances with labels that are not in `class_labels_to_keep`.

    :param annotation: Annotation with scores.
    :param class_labels_to_keep: List of class labels that are to be kept.
    :return: Annotation, where instances with labels that are not in `class_labels_to_keep` have
        been removed.
    """
    labels = annotation["labels"]
    labels = list(labels)

    do_keep = [label in class_labels_to_keep for label in labels]

    annotation = filter_annotation(annotation, do_keep)

    return annotation


def filter_annotation(annotation: Annotation, do_keep: Union[ndarray, List, Tensor]) -> Annotation:
    """Filter instances based on a boolean numpy array.

    :param annotation: Annotation
    :param do_keep: Boolean numpy array or list, which specifies, which instances are to be kept.
    :return: Annotation, where instances with a corresponding false entry in do_keep have been
        removed.
    """

    if np.all(do_keep):
        return annotation

    # TODO: Find a more robust/future-proof solution to determine which entries need to be filtered.
    # IDEA: Try to get number of boxes/masks/labels.

    for key, value in annotation.items():
        if key not in ["image_name", "image_id", "slice_index_x", "slice_index_y"]:
            annotation[key] = annotation[key][do_keep]

    return annotation


def calculate_area_equivalent_diameters(masks: ndarray) -> ndarray:
    """Calculate area equivalent diameters for a numpy array containing masks.

    :param masks: NxHxW numpy array, which stores N instance masks.
    :return: Numpy array of area equivalent diameters.
    """
    masks = np.array(masks).astype(bool)
    masks = masks.reshape(masks.shape[0], -1)
    areas = np.asarray(masks).sum(axis=1)
    return np.sqrt(4 * areas / np.pi)


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


def calculate_minimum_feret_diameters(masks: ndarray) -> ndarray:
    """Calculates the minimum Feret diameters of a numpy array containing masks.

    :param masks: NxHxW numpy array, which stores N instance masks.
    :return: Numpy array of minimum Feret diameters of the masks.
    """
    return np.asarray([calculate_minimum_feret_diameter(mask) for mask in masks])


def calculate_maximum_feret_diameters(masks: ndarray) -> ndarray:
    """Calculates the maximum Feret diameters of a numpy array containing masks.

    :param masks: NxHxW numpy array, which stores N instance masks.
    :return: Numpy array of maximum Feret diameters of the masks.
    """
    return np.asarray([calculate_maximum_feret_diameter(mask) for mask in masks])

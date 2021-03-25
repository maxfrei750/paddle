from typing import Dict, List

import numpy as np
import torch

from paddle.custom_types import Mask


def extract_bounding_boxes(masks: List[Mask]) -> np.ndarray:
    """Extract the bounding boxes of multiple masks.

    :param masks: List of N masks (HxW numpy arrays).
    :return: Nx4 numpy array of bounding boxes.
    """
    boxes = list()
    for mask in masks:
        box = extract_bounding_box(mask)
        boxes.append(box)

    boxes = np.asarray(boxes)

    return boxes


def extract_bounding_box(mask: Mask) -> np.ndarray:
    """Extract the bounding box of a mask.

    :param mask: HxW numpy array
    :return: bounding box
    """
    pos = np.where(mask)  # TODO: Check if np.nonzero can be used instead

    if not (pos[0].size or pos[1].size):
        return np.array([0, 0, 0, 0])

    xmin = np.min(pos[1])
    xmax = np.max(pos[1]) + 1
    ymin = np.min(pos[0])
    ymax = np.max(pos[0]) + 1
    return np.array([xmin, ymin, xmax, ymax])


def dictionary_to_cpu(dictionary: Dict):
    """Move dictionary values to the cpu, if they support it.

    :param dictionary: Dictionary, whose values are to be moved to the cpu.
    :return: Dictionary, with values moved to cpu (if they support it).
    """
    for key, value in dictionary.items():
        if callable(getattr(value, "cpu", None)):
            dictionary[key] = value.cpu()

    return dictionary


def dictionary_to_device(dictionary: Dict, device: torch.device):
    """Move dictionary values to a torch device, if they support it.

    :param dictionary: Dictionary, whose values are to be moved to the torch device.
    :param device: torch device
    :return: Dictionary, with values moved to torch device (if they support it).
    """
    for key, value in dictionary.items():
        if callable(getattr(value, "to", None)):
            dictionary[key] = value.to(device)

    return dictionary

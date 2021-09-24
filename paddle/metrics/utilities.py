from typing import Literal, Optional

import torch
from torch import Tensor
from torchvision.ops import box_iou


def mask_iou(
    masks_prediction: Tensor,
    masks_target: Tensor,
    boxes_prediction: Tensor,
    boxes_target: Tensor,
) -> Tensor:
    """Calculates IoU matrix, based on instance masks.

    :param masks_prediction: NxHxW Tensor which holds predicted masks.
    :param masks_target: NxHxW Tensor which holds target masks.
    :param boxes_prediction: Nx4 Tensor which holds predicted boxes.
    :param boxes_target: Nx4 Tensor which holds target boxes.
    :return: NxM tensor containing the pairwise IoU values.
    """

    # TODO: Support detections without bounding boxes by calculating bounding boxes based on masks.

    box_iou_matrix = box_iou(boxes_prediction, boxes_target)  # predictions x targets
    mask_iou_matrix = torch.zeros_like(box_iou_matrix)

    masks_prediction = torch.round(masks_prediction).bool()
    masks_target = masks_target.bool()

    for p, (mask_prediction, box_prediction) in enumerate(zip(masks_prediction, boxes_prediction)):

        for t, (mask_target, box_target) in enumerate(zip(masks_target, boxes_target)):
            # Only calculate mask iou, if boxes overlap.
            if box_iou_matrix[p, t]:
                x_corners = torch.cat([box_target[0::2], box_prediction[0::2]])
                y_corners = torch.cat([box_target[1::2], box_prediction[1::2]])

                x_min = torch.floor(x_corners.min()).int()
                y_min = torch.floor(y_corners.min()).int()
                x_max = torch.ceil(x_corners.max()).int()
                y_max = torch.ceil(y_corners.max()).int()

                mask_overlap_target = mask_target[y_min:y_max, x_min:x_max]
                mask_overlap_prediction = mask_prediction[y_min:y_max, x_min:x_max]

                area_intersection = torch.logical_and(
                    mask_overlap_target, mask_overlap_prediction
                ).sum()
                area_union = torch.logical_or(mask_overlap_target, mask_overlap_prediction).sum()

                iou = area_intersection / area_union

                mask_iou_matrix[p, t] = iou

    return mask_iou_matrix


def calculate_iou_matrix(
    boxes_predicted: Tensor,
    boxes_target: Tensor,
    iou_type: Literal["box", "mask"],
    masks_predicted: Optional[Tensor] = None,
    masks_target: Optional[Tensor] = None,
):
    """Calculate the Intersections over Unions (IOUs) of N predictions with M targets.
        Supports both box and mask IOU.


    :param boxes_predicted: Bounding boxes of predictions (Tensor[Nx4]).
    :param boxes_target: Bounding boxes of targets (Tensor[Mx4]).
    :param iou_type: Either "mask" or "box". Controls which kind of IOU is calculated.
    :param masks_predicted: Masks of predictions (Tensor[NxHxW]). Only needed if `iou_type` is
        "mask".
    :param masks_target: Masks of targets (Tensor[MxHxW). Only needed if `iou_type` is "mask".
    :return: Intersections over Unions (Tensor[NxM])
    """
    if iou_type == "box":
        ious = box_iou(boxes_predicted, boxes_target)
    elif iou_type == "mask":

        if masks_target is None or boxes_target is None:
            raise ValueError(
                "`masks_target` and `boxes_target` are required, if `iou_type` is 'mask'."
            )

        ious = mask_iou(
            masks_predicted,
            masks_target,
            boxes_predicted,
            boxes_target,
        )
    else:
        raise ValueError(f"Unknown iou_type: {iou_type}. Expected 'box' or 'mask'.")

    return ious

import torch
from torch import Tensor
from torchvision.ops import box_iou

from ..custom_types import ArrayLike


def mask_iou(
    masks_prediction: ArrayLike,
    masks_target: ArrayLike,
    boxes_prediction: ArrayLike,
    boxes_target: ArrayLike,
) -> Tensor:
    """Calculates IoU matrix, based on instance masks.

    :param masks_prediction: NxHxW Tensor which holds predicted masks.
    :param masks_target: NxHxW Tensor which holds target masks.
    :param boxes_prediction: Nx4 Tensor which holds predicted boxes.
    :param boxes_target: Nx4 Tensor which holds target boxes.
    :return: NxM tensor containing the pairwise IoU values.
    """

    # TODO: Support detections without bounding boxes by calculating bounding boxes based on
    #  masks.

    box_iou_matrix = box_iou(boxes_prediction, boxes_target)  # predictions x targets
    mask_iou_matrix = torch.zeros_like(box_iou_matrix)

    masks_prediction = torch.round(masks_prediction.squeeze(dim=0)).bool()
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

                mask_iou = area_intersection / area_union

                mask_iou_matrix[p, t] = mask_iou

    return mask_iou_matrix

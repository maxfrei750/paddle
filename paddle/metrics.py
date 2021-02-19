from typing import Iterable, List, Literal, Tuple

import torch
from pytorch_lightning.metrics import Metric
from torch import tensor
from torchvision.ops.boxes import box_iou

from .custom_types import Annotation

# TODO: Add multi-class support.


class AveragePrecision(Metric):
    """Average Precision (AP) Metric

    AP calculation based on:
        https://www.kaggle.com/kshitijpatil09/pytorch-mean-absolute-precision-calculation?scriptVersionId=40848029
    :param iou_thresholds: Detections with an IoU below the specified IoU thresholds are discarded. If
        ``iou_thresholds`` has more than one element, then the mean average precision (mAP) at these different
        thresholds is calculated.
    :param iou_type: Defines, which attribute of the detected objects is used to calculate the IoU.
        "box"  - bounding box, default
        "mask" - mask
    :param box_format: Bounding box format
        "pascal_voc" - [xmin, ymin, xmax, ymax]
        "coco"       - [xmin, ymin, w, h]
    :param dist_sync_on_step: Synchronize metric state across processes at each ``forward()`` before returning the
        value at the step.
    """

    def __init__(
        self,
        iou_thresholds: Iterable,
        iou_type: Literal["box", "mask"],
        box_format: Literal["coco", "pascal_voc"] = "pascal_voc",
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.iou_thresholds = iou_thresholds
        self.iou_type = iou_type
        self.box_format = box_format

        self.add_state("average_precision_sum", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_samples", default=tensor(0.0), dist_reduce_fx="sum")

    # noinspection PyMethodOverriding
    def update(self, predictions: List[Annotation], targets: Tuple[Annotation, ...]) -> None:
        """Add new data for the calculation of the metric.

        :param predictions: List of dictionaries with prediction data, such as boxes and masks.
        :param targets: Tuple of dictionaries with target data, such as boxes and masks.
        """
        for prediction, target in zip(predictions, targets):
            self.num_samples += 1
            self.average_precision_sum += self._calculate_average_precision(prediction, target)

    def compute(self) -> float:
        """Computes metric based on the gathered data."""
        return self.average_precision_sum / self.num_samples

    def _calculate_average_precision(self, prediction: Annotation, target: Annotation):
        """Calculate average precision of a set of detections (e.g. from a single image).

        Based on:
            https://www.kaggle.com/kshitijpatil09/pytorch-mean-absolute-precision-calculation?scriptVersionId=40848029

        :param prediction: Dictionary with prediction data. Must include boxes or masks.
        :param target: Dictionary with target data. Must include boxes or masks.
        """

        prediction = self._sort_by_score(prediction)
        iou_matrix = self._calculate_iou_matrix(prediction, target)

        average_precisions = []

        for iou_threshold in self.iou_thresholds:
            filtered_iou_matrix = self._apply_threshold(iou_matrix, iou_threshold)
            mappings = self._get_mappings(filtered_iou_matrix)

            # Mean Average Precision calculation
            true_positives = mappings.sum()
            false_positives = mappings.sum(0).eq(0).sum()
            false_negatives = mappings.sum(1).eq(0).sum()
            average_precision = true_positives / (
                true_positives + false_positives + false_negatives
            )

            average_precisions.append(average_precision)

        return sum(average_precisions) / len(average_precisions)

    def _calculate_iou_matrix(self, prediction: Annotation, target: Annotation) -> torch.Tensor:
        """Calculates a matrix containing the pairwise IoU values for every box in ``prediction`` and ``target``.

        :param prediction: Dictionary with prediction data. Must include boxes or masks.
        :param target: Dictionary with target data. Must include boxes or masks.
        :return: NxM tensor containing the pairwise IoU values.
        """
        if self.iou_type == "box":
            return self._calculate_box_iou_matrix(prediction, target)
        elif self.iou_type == "mask":
            return self._calculate_mask_iou_matrix(prediction, target)
        else:
            raise ValueError(f"Unknown iou_type: {self.iou_type}")

    def _calculate_mask_iou_matrix(
        self, prediction: Annotation, target: Annotation
    ) -> torch.Tensor:
        """Calculates IoU matrix, based on instance masks.

        :param prediction: Dictionary with prediction data. Must include boxes and masks.
        :param target: Dictionary with target data. Must include boxes and masks.
        :return: NxM tensor containing the pairwise IoU values.
        """

        # TODO: Support detections without bounding boxes by calculating bounding boxes based on
        #  masks.

        box_iou_matrix = self._calculate_box_iou_matrix(prediction, target)  # predictions x targets
        mask_iou_matrix = torch.zeros_like(box_iou_matrix)

        prediction_boxes = prediction["boxes"].clone()
        target_boxes = target["boxes"].clone()

        prediction_masks = torch.round(prediction["masks"].clone().squeeze()).bool()
        target_masks = target["masks"].clone().bool()

        if self.box_format == "coco":
            target_boxes = self._coco_to_pascal_voc(target_boxes)
            prediction_boxes = self._coco_to_pascal_voc(prediction_boxes)

        for p, (prediction_mask, prediction_box) in enumerate(
            zip(prediction_masks, prediction_boxes)
        ):

            for t, (target_mask, target_box) in enumerate(zip(target_masks, target_boxes)):

                # Only calculate mask iou, if boxes overlap.
                if box_iou_matrix[p, t]:
                    x_corners = torch.cat([target_box[0::2], prediction_box[0::2]])
                    y_corners = torch.cat([target_box[1::2], prediction_box[1::2]])

                    x_min = torch.floor(x_corners.min()).int()
                    y_min = torch.floor(y_corners.min()).int()
                    x_max = torch.ceil(x_corners.max()).int()
                    y_max = torch.ceil(y_corners.max()).int()

                    target_mask_overlap = target_mask[y_min:y_max, x_min:x_max]
                    prediction_mask_overlap = prediction_mask[y_min:y_max, x_min:x_max]

                    area_intersection = torch.logical_and(
                        target_mask_overlap, prediction_mask_overlap
                    ).sum()
                    area_union = torch.logical_or(
                        target_mask_overlap, prediction_mask_overlap
                    ).sum()

                    mask_iou = area_intersection / area_union

                    mask_iou_matrix[p, t] = mask_iou

        return mask_iou_matrix

    def _calculate_box_iou_matrix(self, prediction: Annotation, target: Annotation) -> torch.Tensor:
        """Calculates IoU matrix, based on instance bounding boxes.

        Based on:
            https://www.kaggle.com/kshitijpatil09/pytorch-mean-absolute-precision-calculation?scriptVersionId=40848029

        :param prediction: Dictionary with prediction data. Must include boxes and masks.
        :param target: Dictionary with target data. Must include boxes and masks.
        :return: NxM tensor containing the pairwise IoU values.
        """
        target_boxes = target["boxes"]
        prediction_boxes = prediction["boxes"]

        if self.box_format == "coco":
            target_boxes = self._coco_to_pascal_voc(target_boxes)
            prediction_boxes = self._coco_to_pascal_voc(prediction_boxes)

        target_boxes = self._align_coordinates(target_boxes)
        prediction_boxes = self._align_coordinates(prediction_boxes)

        return box_iou(prediction_boxes, target_boxes)

    @staticmethod
    def _apply_threshold(iou_matrix: torch.Tensor, iou_threshold: float):
        """Set elements below ``iou_thresholds`` to zero.

        :param iou_matrix: NxM tensor containing pairwise IoU values
        :param iou_threshold: elements of ``iou_matrix`` below this value are set to zero
        :return: IoU matrix, with elements below the IoU threshold set to zero.
        """
        device = iou_matrix.device
        iou_threshold = tensor(iou_threshold, device=device)
        iou_matrix = iou_matrix.where(iou_matrix > iou_threshold, tensor(0.0, device=device))
        return iou_matrix

    @staticmethod
    def _sort_by_score(prediction: Annotation) -> Annotation:
        """Sort prediction masks and bounding boxes by score (descending).

        :param prediction: Dictionary with prediction data. Must include scores.
        :return: Annotation with boxes and/or masks sorted based on their score.
        """
        order = prediction["scores"].argsort(descending=True)

        for key in ["boxes", "masks"]:
            if key in prediction:
                prediction[key] = prediction[key].clone()[order]

        return prediction

    @staticmethod
    def _coco_to_pascal_voc(boxes: torch.Tensor) -> torch.Tensor:
        """Convert bounding boxes from Pascal VOC  to COCO format.
            Pascal VOC format - [xmin, ymin, xmax, ymax]
            COCO format       - [xmin, ymin, w, h]

        Based on:
            https://www.kaggle.com/kshitijpatil09/pytorch-mean-absolute-precision-calculation?scriptVersionId=40848029

        :param boxes: Nx4 tensor in COCO format
        :return: Nx4 tensor in Pascal VOC format
        """
        boxes = boxes.clone()
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return boxes

    @staticmethod
    def _align_coordinates(boxes: torch.Tensor) -> torch.Tensor:
        """Align coordinates (x1,y1) < (x2,y2) to work with the torchvision ``box_iou`` operation.

        Based on:
            https://www.kaggle.com/kshitijpatil09/pytorch-mean-absolute-precision-calculation?scriptVersionId=40848029

        :param boxes: Nx4 tensor in Pascal VOC format ([xmin, ymin, xmax, ymax])
        :return: Nx4 tensor storing aligned boxes.
        """

        x1y1 = torch.min(
            boxes[
                :,
                :2,
            ],
            boxes[:, 2:],
        )
        x2y2 = torch.max(
            boxes[
                :,
                :2,
            ],
            boxes[:, 2:],
        )
        boxes = torch.cat([x1y1, x2y2], dim=1)
        return boxes

    @staticmethod
    def _get_mappings(iou_matrix: torch.Tensor) -> torch.Tensor:
        """
        Based on:
            https://www.kaggle.com/kshitijpatil09/pytorch-mean-absolute-precision-calculation?scriptVersionId=40848029

        :param iou_matrix: NxM tensor containing pairwise IoU values
        :return: Mapping of prediction instances and target instances.
        """
        mappings = torch.zeros_like(iou_matrix)
        _, pr_count = iou_matrix.shape

        # first mapping (max iou for first pred_box)
        if not iou_matrix[:, 0].eq(0.0).all():
            # if not a zero column
            mappings[iou_matrix[:, 0].argsort()[-1], 0] = 1

        for pr_idx in range(1, pr_count):
            # Sum of all the previous mapping columns will let
            # us know which target_boxes-boxes are already assigned
            not_assigned = torch.logical_not(mappings[:, :pr_idx].sum(1)).long()

            # Considering unassigned target_boxes-boxes for further evaluation
            targets = not_assigned * iou_matrix[:, pr_idx]

            # If no target_boxes-box satisfy the previous conditions
            # for the current pred-box, ignore it (False Positive)
            if targets.eq(0).all():
                continue

            # max-iou from current column after all the filtering
            # will be the pivot element for mapping
            pivot = targets.argsort()[-1]
            mappings[pivot, pr_idx] = 1
        return mappings

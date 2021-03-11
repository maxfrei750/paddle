from typing import Any, List, Literal, Optional, Tuple

import torch
from pytorch_lightning.metrics.classification.confusion_matrix import (
    ConfusionMatrix as ConfusionMatrixBase,
)
from torch import Tensor

from ..custom_types import Annotation
from ..visualization import plot_confusion_matrix
from .utilities import calculate_iou_matrix


class ConfusionMatrix(ConfusionMatrixBase):
    """Confusion matrix metric for object detection.

    :param num_classes: Number of classes in the dataset (including the background class).
    :param iou_type: Type of Intersetion over Union (IOU) used to determine if a prediction matches
        a target. Either "box" or "mask".
    :param iou_threshold: IOU threshold, above which a prediction is considered a match for a
        target.
    :param score_threshold: Score threshold, above which a prediction is considered non-background.
        default: 0.5
    :param normalize: Normalization mode for confusion matrix. Choose from
            - ``None``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
    :param compute_on_step: Forward only calls ``update()`` and return None if this is set to False.
        default: True
    :param dist_sync_on_step: Synchronize metric state across processes at each ``forward()``
            before returning the value at the step. default: False
    :param process_group: Specify the process group on which synchronization is called.
        default: None (which selects the entire world)
    """

    def __init__(
        self,
        num_classes: int,
        iou_type: Literal["box", "mask"],
        iou_threshold: float,
        score_threshold: float = 0.5,
        normalize: Optional[str] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__(
            num_classes, normalize, 1, compute_on_step, dist_sync_on_step, process_group
        )

        self.iou_threshold = iou_threshold
        self.iou_type = iou_type
        self.score_threshold = score_threshold

        # Remove ambiguous attribute from parent class.
        delattr(self, "threshold")

    def update(self, predictions: List[Annotation], targets: Tuple[Annotation, ...]) -> None:
        """Updates the confusion matrix based on the supplied targets and predictions.

        :param predictions: List of dictionaries with prediction data, such as boxes and masks.
        :param targets: Tuple of dictionaries with target data, such as boxes and masks.
        """

        for prediction, target in zip(predictions, targets):
            confusion_matrix = self._evaluate_image(prediction, target)

            self.confmat += confusion_matrix

    def _evaluate_image(self, prediction: Annotation, target: Annotation) -> Tensor:
        """Evaluates the target and prediction instances of a single image.

        :param prediction: Dictionary with prediction data, such as boxes and masks.
        :param target: Dictionary with target data, such as boxes and masks.
        :return:
        """

        device = target["boxes"].device

        boxes_pred = prediction["boxes"]
        boxes_gt = target["boxes"]

        labels_pred = prediction["labels"]
        labels_gt = target["labels"]

        scores_pred = prediction["scores"]

        if self.iou_type == "mask":
            masks_pred = prediction["masks"]
            masks_gt = target["masks"]
        else:
            masks_pred = [None] * len(boxes_pred)
            masks_gt = [None] * len(boxes_gt)

        descending_score_indices = torch.argsort(scores_pred, descending=True)
        is_assigned_gt = torch.zeros_like(labels_gt, dtype=torch.bool)

        confusion_matrix = torch.zeros(self.num_classes, self.num_classes, device=device)

        # Iterate all predictions.
        for box_pred, label_pred, mask_pred, score_pred in zip(
            boxes_pred[descending_score_indices],
            labels_pred[descending_score_indices],
            masks_pred[descending_score_indices],
            scores_pred[descending_score_indices],
        ):
            ious = calculate_iou_matrix(
                torch.unsqueeze(box_pred, dim=0),
                boxes_gt,
                self.iou_type,
                mask_pred,
                masks_gt,
            )

            # Assign predictions with a score below score_threshold to the background class.
            if score_pred <= self.score_threshold:
                label_pred = 0

            best_iou, best_gt_index = ious.squeeze(0).max(0)

            if best_iou > self.iou_threshold and not is_assigned_gt[best_gt_index]:
                # We have a match, so the predicted label should be that of the matching ground
                # truth.
                label_gt = labels_gt[best_gt_index]

                # mark the ground truth with the highest iou as assigned
                is_assigned_gt[best_gt_index] = True
            else:
                # We don't have a matching ground truth, so the predicted label should have been
                # that of the background class (0)
                label_gt = 0

            confusion_matrix[label_gt, label_pred] += 1

        # Iterate all ground truths that where not detected.
        label_pred = 0  # background class
        for label_gt in labels_gt[~is_assigned_gt]:
            confusion_matrix[label_gt, label_pred] += 1

        return confusion_matrix

    def plot(self, class_names: Optional[List[str]] = None):
        """Compute and plot the confusion matrix.

        :param class_names: Optional class names to be used as labels.
        :return: figure handle
        """
        return plot_confusion_matrix(self.compute(), class_names=class_names)

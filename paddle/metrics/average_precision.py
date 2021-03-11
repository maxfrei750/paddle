from collections import Counter
from typing import List, Literal, Optional, Tuple, Union

import torch
from pytorch_lightning.metrics import Metric
from torch import Tensor, tensor

from ..custom_types import Annotation, ArrayLike
from .utilities import calculate_iou_matrix


class AveragePrecision(Metric):
    """Average Precision (AP) Metric

    :param iou_type: Defines, which attribute of the detected objects is used to calculate the IoU.
        "box"  - bounding box
        "mask" - mask
    :param iou_thresholds: threshold for IoU score for determining true positive and
        false positive predictions.
    :param ap_calculation_type: method to calculate the average precision of the precision-recall
        curve
        - `'step'`: calculate the step function integral, the same way as
        :func:`~pytorch_lightning.metrics.functional.average_precision.average_precision`
        - `'VOC2007'`: calculate the 11-point sampling of interpolation of the precision recall
        curve
        - `'VOC2010'`: calculate the step function integral of the interpolated precision recall
        curve
        - `'COCO'`: calculate the 101-point sampling of the interpolated precision recall curve
    :param dist_sync_on_step: Synchronize metric state across processes at each `forward()` before
        returning the value at the step.
    """

    def __init__(
        self,
        num_foreground_classes: int,
        iou_thresholds: Union[ArrayLike, List],
        iou_type: Literal["box", "mask"],
        ap_calculation_type: Literal["step", "VOC2007", "VOC2010", "COCO"],
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.iou_thresholds = iou_thresholds
        self.iou_type = iou_type
        self.ap_calculation_type = ap_calculation_type
        self.num_foreground_classes = num_foreground_classes
        self.num_iou_thresholds = len(iou_thresholds)

        self.class_labels = list(range(1, self.num_foreground_classes + 1))

        # Mimic a dictionary.
        # TODO: Find a less hacky solution.
        for class_label in self.class_labels:
            self.add_state(f"false_positives_class{class_label}", default=[], dist_reduce_fx="cat")
            self.add_state(f"true_positives_class{class_label}", default=[], dist_reduce_fx="cat")
            self.add_state(
                f"num_targets_class{class_label}", default=tensor(0), dist_reduce_fx="sum"
            )

    # noinspection PyMethodOverriding
    def update(self, predictions: List[Annotation], targets: Tuple[Annotation, ...]) -> None:
        """Add new data for the calculation of the metric.

        :param predictions: List of dictionaries with prediction data, such as boxes and masks.
        :param targets: Tuple of dictionaries with target data, such as boxes and masks.
        """

        device = targets[0]["boxes"].device

        # Create a unique index for every image.
        unique_image_indices = [hash(target["boxes"]) for target in targets]

        num_instances_per_image_prediction = [
            len(prediction["boxes"]) for prediction in predictions
        ]

        num_instances_per_image_target = [len(target["boxes"]) for target in targets]

        image_indices_prediction = torch.cat(
            [
                torch.ones(n, dtype=torch.int64, device=device) * index
                for n, index in zip(num_instances_per_image_prediction, unique_image_indices)
            ]
        )

        image_indices_target = torch.cat(
            [
                torch.ones(n, dtype=torch.int64, device=device) * index
                for n, index in zip(num_instances_per_image_target, unique_image_indices)
            ]
        )

        boxes_prediction = torch.cat([prediction["boxes"] for prediction in predictions])
        boxes_target = torch.cat([target["boxes"] for target in targets])

        labels_prediction = torch.cat([prediction["labels"] for prediction in predictions])
        labels_target = torch.cat([target["labels"] for target in targets])

        scores_prediction = torch.cat([prediction["scores"] for prediction in predictions])

        if self.iou_type == "mask":
            masks_prediction = torch.cat([prediction["masks"] for prediction in predictions])
            masks_target = torch.cat([target["masks"] for target in targets])
        else:
            masks_prediction = None
            masks_target = None

        (
            true_positives_per_class,
            false_positives_per_class,
            num_targets_per_class,
        ) = self._evaluate_batch(
            image_indices_prediction,
            scores_prediction,
            labels_prediction,
            boxes_prediction,
            image_indices_target,
            labels_target,
            boxes_target,
            masks_prediction,
            masks_target,
        )

        for key in true_positives_per_class.keys():
            getattr(self, f"true_positives_class{key}").append(true_positives_per_class[key])
            getattr(self, f"false_positives_class{key}").append(false_positives_per_class[key])
            setattr(
                self,
                f"num_targets_class{key}",
                getattr(self, f"num_targets_class{key}") + num_targets_per_class[key],
            )

    def compute(self) -> Tensor:
        """Computes metric based on the gathered data."""

        average_precisions = torch.zeros(self.num_foreground_classes, len(self.iou_thresholds))

        for class_index, c in enumerate(self.class_labels):

            tps = torch.cat(getattr(self, f"true_positives_class{c}"))
            fps = torch.cat(getattr(self, f"false_positives_class{c}"))
            num_targets = getattr(self, f"num_targets_class{c}").cpu()

            for iou_idx, _ in enumerate(self.iou_thresholds):
                tps_cum = torch.cumsum(tps[:, iou_idx], dim=0)
                fps_cum = torch.cumsum(fps[:, iou_idx], dim=0)

                precision = tps_cum / (tps_cum + fps_cum)
                recall = tps_cum / num_targets if num_targets else tps_cum
                precision = torch.cat([reversed(precision), torch.tensor([1.0])])
                recall = torch.cat([reversed(recall), torch.tensor([0.0])])
                if self.ap_calculation_type == "step":
                    average_precision = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
                elif self.ap_calculation_type == "VOC2007":
                    average_precision = 0
                    recall_thresholds = torch.linspace(0, 1, 11)
                    for threshold in recall_thresholds:
                        points = precision[:-1][recall[:-1] >= threshold]
                        average_precision += torch.max(points) / 11 if len(points) else 0
                elif self.ap_calculation_type == "VOC2010":
                    for i in range(len(precision)):
                        precision[i] = torch.max(precision[: i + 1])
                    average_precision = -torch.sum((recall[1:] - recall[:-1]) * precision[:-1])
                elif self.ap_calculation_type == "COCO":
                    average_precision = 0
                    recall_thresholds = torch.linspace(0, 1, 101)
                    for threshold in recall_thresholds:
                        points = precision[:-1][recall[:-1] >= threshold]
                        average_precision += torch.max(points) / 101 if len(points) else 0
                else:
                    raise NotImplementedError(f"'{self.ap_calculation_type}' is not supported.")
                average_precisions[class_index, iou_idx] = average_precision
        return torch.mean(average_precisions)

    def _evaluate_batch(
        self,
        pred_image_indices: Tensor,
        prediction_scores: Tensor,
        prediction_labels: Tensor,
        prediction_boxes: Tensor,
        target_image_indices: Tensor,
        target_labels: Tensor,
        target_bboxes: Tensor,
        prediction_masks: Optional[Tensor] = None,
        target_masks: Optional[Tensor] = None,
    ):
        """
            pred_image_indices: an (N,)-shaped Tensor of image indices of the predictions
            prediction_scores: an (N,)-shaped Tensor of probabilities of the predictions
            prediction_labels: an (N,)-shaped Tensor of predicted labels
            prediction_boxes: an (N, 4)-shaped Tensor of predicted bounding boxes
            target_image_indices: an (M,)-shaped Tensor of image indices of the groudn truths
            target_labels: an (M,)-shaped Tensor of ground truth labels
            target_bboxes: an (M, 4)-shaped Tensor of ground truth bounding boxes
        References:
            - host.robots.ox.ac.uk/pascal/VOC/
            - https://ccc.inaoep.mx/~villasen/bib/AN%20OVERVIEW%20OF%20EVALUATION%20METHODS%20IN%20TREC%20AD%20HOC%20IR%20AND%20TREC%20QA.pdf#page=15
            - https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py

        Based on: https://github.com/PyTorchLightning/pytorch-lightning/pull/4564
        """

        true_positives_per_class = {}
        false_positives_per_class = {}
        num_targets_per_class = {}

        class_labels = torch.cat([prediction_labels, target_labels]).unique()

        for class_idx, class_label in enumerate(class_labels):
            # Descending indices w.r.t. class probability for class c
            descending_score_indices = torch.argsort(prediction_scores, descending=True)[
                prediction_labels == class_label
            ]
            # No predictions for this class so average precision is 0
            if len(descending_score_indices) == 0:
                continue
            targets_per_images = Counter(
                [index.item() for index in target_image_indices[target_labels == class_label]]
            )
            targets_assigned = {
                image_index: torch.zeros(count, self.num_iou_thresholds, dtype=torch.bool)
                for image_index, count in targets_per_images.items()
            }
            tps = torch.zeros(len(descending_score_indices), self.num_iou_thresholds)
            fps = torch.zeros(len(descending_score_indices), self.num_iou_thresholds)
            for i, prediction_index in enumerate(descending_score_indices):
                image_index = pred_image_indices[prediction_index].item()
                # Get the ground truth bboxes of class c and the same image index as the prediction
                gt_boxes = target_bboxes[
                    (target_image_indices == image_index) & (target_labels == class_label)
                ]

                pred_box = torch.unsqueeze(prediction_boxes[prediction_index], dim=0)

                if self.iou_type == "mask":
                    gt_masks = target_masks[
                        (target_image_indices == image_index) & (target_labels == class_label)
                    ]
                    pred_mask = prediction_masks[prediction_index]
                else:
                    gt_masks = None
                    pred_mask = None

                ious = calculate_iou_matrix(
                    pred_box,
                    gt_boxes,
                    self.iou_type,
                    pred_mask,
                    gt_masks,
                )

                best_iou, best_target_index = (
                    ious.squeeze(0).max(0) if len(gt_boxes) > 0 else (0, -1)
                )
                # Prediction is a true positive is the IoU score is greater than the threshold and
                # the corresponding ground truth has only one prediction assigned to it
                for iou_index, iou_threshold in enumerate(self.iou_thresholds):
                    if (
                        best_iou > iou_threshold
                        and not targets_assigned[image_index][best_target_index][iou_index]
                    ):
                        targets_assigned[image_index][best_target_index][iou_index] = True
                        tps[i, iou_index] = 1
                    else:
                        fps[i, iou_index] = 1

            num_targets = len(target_labels[target_labels == class_label])

            true_positives_per_class[int(class_label)] = tps
            false_positives_per_class[int(class_label)] = fps
            num_targets_per_class[int(class_label)] = num_targets

        return true_positives_per_class, false_positives_per_class, num_targets_per_class

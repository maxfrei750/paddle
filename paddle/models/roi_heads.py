from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torchvision.models.detection.roi_heads import RoIHeads as RoIHeadsBase
from torchvision.models.detection.roi_heads import (
    fastrcnn_loss,
    keypointrcnn_inference,
    keypointrcnn_loss,
    maskrcnn_inference,
    maskrcnn_loss,
)


class RoIHeads(RoIHeadsBase):
    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
        # Mask
        mask_roi_pool=None,
        mask_head=None,
        mask_predictor=None,
        # Keypoints
        keypoint_roi_pool=None,
        keypoint_head=None,
        keypoint_predictor=None,
        # Fiber Width
        fiber_width_roi_pool=None,
        fiber_width_head=None,
        fiber_width_predictor=None,
        # Fiber Length
        fiber_length_roi_pool=None,
        fiber_length_head=None,
        fiber_length_predictor=None,
    ):
        super().__init__(
            box_roi_pool,
            box_head,
            box_predictor,
            fg_iou_thresh,
            bg_iou_thresh,
            batch_size_per_image,
            positive_fraction,
            bbox_reg_weights,
            score_thresh,
            nms_thresh,
            detections_per_img,
            mask_roi_pool,
            mask_head,
            mask_predictor,
            keypoint_roi_pool,
            keypoint_head,
            keypoint_predictor,
        )

        self.fiber_length_roi_pool = fiber_length_roi_pool
        self.fiber_length_head = fiber_length_head
        self.fiber_length_predictor = fiber_length_predictor

        self.fiber_width_roi_pool = fiber_width_roi_pool
        self.fiber_width_head = fiber_width_head
        self.fiber_width_predictor = fiber_width_predictor

    def has_fiber_width(self):
        if self.fiber_width_roi_pool is None:
            return False
        if self.fiber_width_head is None:
            return False
        if self.fiber_width_predictor is None:
            return False
        return True

    def has_fiber_length(self):
        if self.fiber_length_roi_pool is None:
            return False
        if self.fiber_length_head is None:
            return False
        if self.fiber_length_predictor is None:
            return False
        return True

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]:
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, "target boxes must of float type"
                assert t["labels"].dtype == torch.int64, "target labels must of int64 type"
                if self.has_keypoint():
                    assert (
                        t["keypoints"].dtype == torch.float32
                    ), "target keypoints must of float type"

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                assert matched_idxs is not None
                # during training, only focus on positive boxes
                num_images = len(proposals)
                mask_proposals = []
                pos_matched_idxs = []
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            if self.mask_roi_pool is not None:
                mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.mask_head(mask_features)
                mask_logits = self.mask_predictor(mask_features)
            else:
                mask_logits = torch.tensor(0)
                raise Exception("Expected mask_roi_pool to be not None")

            loss_mask = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None
                assert mask_logits is not None

                gt_masks = [t["masks"] for t in targets]
                gt_labels = [t["labels"] for t in targets]
                rcnn_loss_mask = maskrcnn_loss(
                    mask_logits, mask_proposals, gt_masks, gt_labels, pos_matched_idxs
                )
                loss_mask = {"loss_mask": rcnn_loss_mask}
            else:
                labels = [r["labels"] for r in result]
                masks_probs = maskrcnn_inference(mask_logits, labels)
                for mask_prob, r in zip(masks_probs, result):
                    r["masks"] = mask_prob

            losses.update(loss_mask)

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.keypoint_roi_pool is not None
            and self.keypoint_head is not None
            and self.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                keypoint_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    keypoint_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            keypoint_features = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.keypoint_head(keypoint_features)
            keypoint_logits = self.keypoint_predictor(keypoint_features)

            loss_keypoint = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_keypoints = [t["keypoints"] for t in targets]
                rcnn_loss_keypoint = keypointrcnn_loss(
                    keypoint_logits, keypoint_proposals, gt_keypoints, pos_matched_idxs
                )
                loss_keypoint = {"loss_keypoint": rcnn_loss_keypoint}
            else:
                assert keypoint_logits is not None
                assert keypoint_proposals is not None

                keypoints_probs, kp_scores = keypointrcnn_inference(
                    keypoint_logits, keypoint_proposals
                )
                for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                    r["keypoints"] = keypoint_prob
                    r["keypoints_scores"] = kps

            losses.update(loss_keypoint)

        if self.has_fiber_length():
            fiber_length_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                fiber_length_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    fiber_length_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            fiber_length_features = self.fiber_length_roi_pool(
                features, fiber_length_proposals, image_shapes
            )
            fiber_length_features = self.fiber_length_head(fiber_length_features)
            fiber_length_logits = self.fiber_length_predictor(fiber_length_features)

            loss_fiber_length = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_fiber_length = [t["fiber_length"] for t in targets]
                rcnn_loss_fiber_length = fiber_length_loss(
                    fiber_length_logits, fiber_length_proposals, gt_fiber_length, pos_matched_idxs
                )
                loss_fiber_length = {"loss_fiber_length": rcnn_loss_fiber_length}
            else:
                assert fiber_length_logits is not None
                assert fiber_length_proposals is not None

                fiber_length_probs, fiber_length_scores = fiber_length_inference(
                    fiber_length_logits, fiber_length_proposals
                )
                for fiber_length_prob, fiber_length_score, r in zip(
                    fiber_length_probs, fiber_length_scores, result
                ):
                    r["fiber_length"] = fiber_length_prob
                    r["fiber_length_scores"] = fiber_length_score

            losses.update(loss_fiber_length)

        if self.has_fiber_width():
            fiber_width_proposals = [p["boxes"] for p in result]
            if self.training:
                # during training, only focus on positive boxes
                num_images = len(proposals)
                fiber_width_proposals = []
                pos_matched_idxs = []
                assert matched_idxs is not None
                for img_id in range(num_images):
                    pos = torch.where(labels[img_id] > 0)[0]
                    fiber_width_proposals.append(proposals[img_id][pos])
                    pos_matched_idxs.append(matched_idxs[img_id][pos])
            else:
                pos_matched_idxs = None

            fiber_width_features = self.fiber_width_roi_pool(
                features, fiber_width_proposals, image_shapes
            )
            fiber_width_features = self.fiber_width_head(fiber_width_features)
            fiber_width_logits = self.fiber_width_predictor(fiber_width_features)

            loss_fiber_width = {}
            if self.training:
                assert targets is not None
                assert pos_matched_idxs is not None

                gt_fiber_width = [t["fiber_width"] for t in targets]
                rcnn_loss_fiber_width = fiber_width_loss(
                    fiber_width_logits, fiber_width_proposals, gt_fiber_width, pos_matched_idxs
                )
                loss_fiber_width = {"loss_fiber_width": rcnn_loss_fiber_width}
            else:
                assert fiber_width_logits is not None
                assert fiber_width_proposals is not None

                fiber_width_probs, fiber_width_scores = fiber_width_inference(
                    fiber_width_logits, fiber_width_proposals
                )
                for fiber_width_prob, fiber_width_score, r in zip(
                    fiber_width_probs, fiber_width_scores, result
                ):
                    r["fiber_width"] = fiber_width_prob
                    r["fiber_width_scores"] = fiber_width_score

            losses.update(loss_fiber_width)

        return result, losses

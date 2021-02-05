from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import ReduceLROnPlateau

from custom_types import (
    Annotation,
    Batch,
    Image,
    Loss,
    OptimizerConfiguration,
    PartialLosses,
    TestOutput,
    ValidationOutput,
)
from metrics import AveragePrecision
from utilities import dictionary_to_cpu


class LightningMaskRCNN(pl.LightningModule):
    """Lightning version of the torchvision Mask R-CNN architecture with Stochastic Gradient
        Descent, mAP validation and DropLROnPlateau learning rate scheduler.

    :param num_classes: Number of classes of the Mask R-CNN (including the background, so the
        minimum is 2).
    :param learning_rate: Learning rate of the SGD optimizer.
    :param drop_lr_on_plateau_patience: Patience of the `DropLROnPlateau` learning rate scheduler,
        until it drops the learning rate by a factor of 10.
    :param model_kwargs: Keyword arguments which are given to
        `torchvision.models.detection.maskrcnn_resnet50_fpn` during model creation.
    """

    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 0.005,
        drop_lr_on_plateau_patience: int = 10,
        model_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.drop_lr_on_plateau_patience = drop_lr_on_plateau_patience

        if model_kwargs is None:
            self.model_kwargs = {}
        else:
            self.model_kwargs = model_kwargs

        self.model = self.build_model()

        self.validation_metrics = ModuleDict(
            {
                # "AP50": AveragePrecision(iou_thresholds=(0.5,), iou_type="mask"),
                # "AP75": AveragePrecision(iou_thresholds=(0.75,), iou_type="mask"),
                "mAP": AveragePrecision(iou_thresholds=np.arange(0.5, 1, 0.05), iou_type="mask"),
            }
        )

        self.main_validation_metric_name = "mAP"

    def build_model(self) -> torchvision.models.detection.MaskRCNN:
        """Builds the Mask R-CNN model. Based on
            `torchvision.models.detection.maskrcnn_resnet50_fpn`.

        :return: Mask R-CNN model
        """
        # Load an instance segmentation model pre-trained on COCO.
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True,
            **self.model_kwargs,
        )

        # Replace the pretrained box and the mask heads.
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features_box, self.num_classes
        )

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, self.num_classes
        )

        return model

    def forward(
        self, images: Tuple[Image, ...], targets: Optional[Tuple[Annotation, ...]] = None
    ) -> Union[List[Annotation], PartialLosses]:
        """Forward pass through the Mask R-CNN.

        :param images: Input images.
        :param targets: Ground truth annotations.
        :return:
            During training: partial losses of the Mask R-CNN heads
            During validation and test: predictions
        """
        return self.model(images, targets)

    def training_step(self, batch: Batch, batch_idx: int) -> Loss:
        """Takes a batch, inputs it into the model and retrieves and logs losses of the prediction heads.
            Calculate the sum of losses and return it.

        :param batch: Batch of images and ground truths.
        :param batch_idx: Index of the current batch.
        :return: Sum of the losses of the prediction heads.
        """
        images, targets = batch
        partial_losses = self(images, targets)

        # TODO: Implement loss weights.
        loss = sum(partial_loss for partial_loss in partial_losses.values())

        self.log("train/loss", loss)

        for key, value in partial_losses.items():
            self.log("train/" + key, value)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> ValidationOutput:
        """Take a batch from the validation data set and input its images into the model to
            retrieve the associated predictions. Return predictions and and ground truths for later use.

        :param batch: Batch of images and ground truths.
        :param batch_idx: Index of the current batch.
        :return: Predictions and ground truths.
        """
        images, targets = batch
        predictions = self(images)

        return {"predictions": predictions, "targets": targets}

    def validation_step_end(self, outputs: ValidationOutput) -> None:
        """Calculate and log the validation_metrics.

        :param outputs: Outputs of the validation step.
        """
        for metric_name, metric in self.validation_metrics.items():
            metric(outputs["predictions"], outputs["targets"])
            self.log(f"val/{metric_name}", metric)

            if metric_name == self.main_validation_metric_name:
                self.log("hp_metric", metric)

    def test_step(self, batch: Batch, batch_idx: int) -> TestOutput:
        """Take a batch from the test data set and input its images into the model to
            retrieve the associated predictions. Return predictions and and ground truths for later use.

        :param batch: Batch of images and ground truths.
        :param batch_idx: Index of the current batch.
        :return: Predictions, ground truths and input images.
        """
        images, _ = batch
        predictions = self(images)

        return {"predictions": predictions}

    def configure_optimizers(
        self,
    ) -> OptimizerConfiguration:
        """Configure the SGD optimizer and the ReduceLROnPlateau learning rate scheduler.

        :return: Dictionary with the optimizer, the learning rate scheduler and the name of the
            metric monitored by the learning rate scheduler.
        """

        # TODO: Test adding weight_decay= 0.0005.
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(
                optimizer, mode="max", patience=self.drop_lr_on_plateau_patience
            ),
            "monitor": "val/mAP",
        }

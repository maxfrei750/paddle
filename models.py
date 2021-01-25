import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import ReduceLROnPlateau

from metrics import AveragePrecision

# TODO: Docstrings
# TODO: Use typing.
# TODO: Check if validation_step_end can be integrated into validation_step, if multiple gpus are used.
# TODO: Test adding weight_decay= 0.0005 to SGD.


class LightningMaskRCNN(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        learning_rate=0.005,
        drop_lr_on_plateau_patience=10,
        model_kwargs=None,
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

        self.model = self.get_model()

        self.validation_metrics = ModuleDict(
            {
                # "AP50": AveragePrecision(iou_thresholds=(0.5,), iou_type="mask"),
                # "AP75": AveragePrecision(iou_thresholds=(0.75,), iou_type="mask"),
                "mAP": AveragePrecision(iou_thresholds=np.arange(0.5, 1, 0.05), iou_type="mask"),
            }
        )

        self.map_label_to_class_name = None

    def get_model(self):
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

    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        partial_losses = self(images, targets)
        loss = sum(partial_loss for partial_loss in partial_losses.values())

        self.log("train/loss", loss)

        for key, value in partial_losses.items():
            self.log("train/" + key, value)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self(images, targets)

        return {"predictions": predictions, "targets": targets}

    def validation_step_end(self, outputs):
        for metric_name, metric in self.validation_metrics.items():
            metric(outputs["predictions"], outputs["targets"])
            self.log(f"val/{metric_name}", metric)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        return {
            "optimizer": optimizer,
            "lr_scheduler": ReduceLROnPlateau(
                optimizer, mode="max", patience=self.drop_lr_on_plateau_patience
            ),
            "monitor": "val/mAP",
        }

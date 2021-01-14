from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from metrics import AveragePrecision


def get_model(num_classes, num_detections_per_image_max=100):
    model = get_mask_rcnn_resnet50_model(
        num_classes, n_detections_per_image_max=num_detections_per_image_max
    )

    # TODO: Replace with
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    #     num_classes=num_classes,
    #     pretrained=True,
    #     trainable_backbone_layers=config["model"]["trainable_backbone_layers"],
    # )

    return model


def get_mask_rcnn_resnet50_model(num_classes, pretrained=True, n_detections_per_image_max=100):
    # Load an instance segmentation model pre-trained on COCO.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=pretrained, box_detections_per_img=n_detections_per_image_max
    )
    # Get the number of input features for the classifier.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Get the number of input features for the mask classifier.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one.
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# TODO: Docstrings
# TODO: Use typing.
# TODO: Log images
# TODO: Test native pytorch vision maskrcnn
# TODO: Add Hyperparameter: num_detections_max
# TODO: Test Adam
# TODO: Test drop lr on plateau


class LightningMaskRCNN(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        learning_rate=0.005,
        learning_rate_step_size=30,
        learning_rate_drop_factor=0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(num_classes=num_classes)

        self.validation_metrics = ModuleDict(
            {
                # "AP50": AveragePrecision(iou_thresholds=(0.5,), iou_type="mask"),
                # "AP75": AveragePrecision(iou_thresholds=(0.75,), iou_type="mask"),
                "mAP": AveragePrecision(iou_thresholds=np.arange(0.5, 1, 0.05), iou_type="mask"),
            }
        )

        self.learning_rate = learning_rate
        self.learning_rate_step_size = learning_rate_step_size
        self.learning_rate_drop_factor = learning_rate_drop_factor

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

        scheduler = {
            "scheduler": ExponentialLR(optimizer, gamma=self.learning_rate_drop_factor),
            "interval": "epoch",
            "frequency": self.learning_rate_step_size,
            "strict": True,
        }

        return [optimizer], [scheduler]


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning import loggers as pl_loggers
    from pytorch_lightning.callbacks import LearningRateMonitor

    from data import MaskRCNNDataModule

    data_root = Path("data") / "sem"
    log_root = "lightning_logs"
    max_epochs = 100
    cropping_rectangle = (0, 0, 1280, 896)
    fast_dev_run = False
    batch_size = 1
    gpus = 1
    random_seed = 42

    pl.seed_everything(random_seed)

    data_module = MaskRCNNDataModule(
        data_root=data_root,
        cropping_rectangle=cropping_rectangle,
        batch_size=batch_size,
    )

    learning_rate_monitor = LearningRateMonitor()
    tensorboard_logger = pl_loggers.TensorBoardLogger(log_root)

    maskrcnn = LightningMaskRCNN()
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        callbacks=[learning_rate_monitor],
        logger=tensorboard_logger,
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(maskrcnn, datamodule=data_module)

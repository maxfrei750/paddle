from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from callbacks import ExampleDetectionMonitor
from metrics import AveragePrecision

# TODO: Docstrings
# TODO: Use typing.
# TODO: Test Adam
# TODO: Test cyclical learning rate.
# TODO: Check if validation_step_end can be integrated into validation_step, if multiple gpus are used.
# TODO: Add script arguments.
# TODO: Move training script.
# TODO: Optional: Add configs.
# TODO: Test adding weight_decay= 0.0005 to SGD.
# TODO: Add checkpointer: https://pytorch-lightning.readthedocs.io/en/stable/weights_loading.html#automatic-saving


class LightningMaskRCNN(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        num_detections_per_image_max=100,
        learning_rate=0.005,
        learning_rate_scheduler_patience=10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.num_detections_per_image_max = num_detections_per_image_max

        self.model = self.get_model()

        self.validation_metrics = ModuleDict(
            {
                # "AP50": AveragePrecision(iou_thresholds=(0.5,), iou_type="mask"),
                # "AP75": AveragePrecision(iou_thresholds=(0.75,), iou_type="mask"),
                "mAP": AveragePrecision(iou_thresholds=np.arange(0.5, 1, 0.05), iou_type="mask"),
            }
        )

        self.learning_rate = learning_rate
        self.learning_rate_scheduler_patience = learning_rate_scheduler_patience

    def get_model(self):
        # Load an instance segmentation model pre-trained on COCO.
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, box_detections_per_img=self.num_detections_per_image_max
        )

        # Replace the pretrained box and the mask heads.
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, self.num_classes)

        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
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
                optimizer, mode="max", patience=self.learning_rate_scheduler_patience
            ),
            "monitor": "val/mAP",
        }


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

    from data import MaskRCNNDataModule

    find_optimum_learning_rate = False

    data_root = Path("data") / "sem"
    log_root = "lightning_logs"
    max_epochs = 100
    cropping_rectangle = (0, 0, 1280, 896)
    fast_dev_run = False
    batch_size = 8
    gpus = -1
    random_seed = 42
    learning_rate = 0.005
    learning_rate_scheduler_patience = 10
    early_stopping_patience = 20

    pl.seed_everything(random_seed)

    data_module = MaskRCNNDataModule(
        data_root=data_root,
        cropping_rectangle=cropping_rectangle,
        batch_size=batch_size,
    )

    model = LightningMaskRCNN(
        learning_rate=learning_rate,
        learning_rate_scheduler_patience=learning_rate_scheduler_patience,
    )

    if find_optimum_learning_rate:
        lr_tuner = pl.Trainer(auto_lr_find=True, gpus=gpus)
        lr_tuner.tune(model, datamodule=data_module)
        exit()

    callbacks = [
        ModelCheckpoint(monitor="val/mAP", mode="max"),
        EarlyStopping(monitor="val/mAP", mode="max", patience=early_stopping_patience),
        LearningRateMonitor(),
        ExampleDetectionMonitor(),
    ]

    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=TensorBoardLogger(log_root),
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, datamodule=data_module)

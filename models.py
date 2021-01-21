import logging
import os
import warnings
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import ModuleDict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from callbacks import ExampleDetectionMonitor
from data import MaskRCNNDataModule
from metrics import AveragePrecision

# TODO: Disable hydra logging and store config manually
# TODO: Docstrings
# TODO: Use typing.
# TODO: Check if validation_step_end can be integrated into validation_step, if multiple gpus are used.
# TODO: Move training script.
# TODO: Test adding weight_decay= 0.0005 to SGD.
from utilities import get_time_stamp


class LightningMaskRCNN(pl.LightningModule):
    def __init__(
        self,
        num_classes=2,
        num_detections_per_image_max=100,
        learning_rate=0.005,
        drop_lr_on_plateau_patience=10,
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
        self.drop_lr_on_plateau_patience = drop_lr_on_plateau_patience

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
                optimizer, mode="max", patience=self.drop_lr_on_plateau_patience
            ),
            "monitor": "val/mAP",
        }


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="maskrcnn")
def train(config: DictConfig) -> None:
    log_dir = Path.cwd()
    log_root = log_dir.parent
    version = log_dir.name

    os.chdir(hydra.utils.get_original_cwd())

    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(config)}")

    pl.seed_everything(config.program.random_seed)

    data_module = MaskRCNNDataModule(**config.datamodule)

    model = LightningMaskRCNN(**config.model)

    if config.program.search_optimum_learning_rate:
        lr_tuner = pl.Trainer(auto_lr_find=True)
        lr_tuner.tune(model, datamodule=data_module)
        exit()

    callbacks = [
        ModelCheckpoint(monitor="val/mAP", mode="max"),
        EarlyStopping(
            monitor="val/mAP", mode="max", patience=config.callbacks.early_stopping_patience
        ),
        LearningRateMonitor(),
        ExampleDetectionMonitor(),
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=str(log_root), name="", version=version),
        **config.trainer,
    )

    # TODO: Remove this filter, as soon as torchvision>0.8.2 is released.
    warnings.filterwarnings(
        "ignore",
        message="The default behavior for interpolate/upsample with float scale_factor changed",
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()

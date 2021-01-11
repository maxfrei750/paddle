from pathlib import Path

import pytorch_lightning as pl
import torch
import torchvision
from torch.optim.lr_scheduler import ExponentialLR
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from lr_schedulers import LearningRateWarmup


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


class LightningMaskRCNN(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=0.005):
        super().__init__()
        self.model = get_model(num_classes=num_classes)

        self.learning_rate = learning_rate
        self.learning_rate_step_size = 30
        self.gamma = 0.1
        self.warmup_factor = 0.001
        self.warmup_duration = 1000

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        partial_losses = self.model(images, targets)
        loss = sum(partial_loss for partial_loss in partial_losses.values())

        self.log("loss", loss)

        for key, value in partial_losses.items():
            self.log(key, value)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        schedulers = [
            {
                "scheduler": LearningRateWarmup(
                    optimizer,
                    warmup_duration=self.warmup_duration,
                    warmup_factor=self.warmup_factor,
                ),
                "interval": "step",
                "frequency": 1,
                "strict": True,
            },
            {
                "scheduler": ExponentialLR(optimizer, gamma=self.gamma),
                "interval": "epoch",
                "frequency": self.learning_rate_step_size,
                "strict": True,
            },
        ]

        return [optimizer], schedulers


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import LearningRateMonitor

    from data import MaskRCNNDataModule

    cropping_rectangle = (0, 0, 1280, 896)

    data_root = Path("data") / "tem"

    data_module = MaskRCNNDataModule(
        data_root=data_root, cropping_rectangle=cropping_rectangle, batch_size=1
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    maskrcnn = LightningMaskRCNN()

    trainer = pl.Trainer(gpus=1, max_epochs=100, callbacks=[lr_monitor])
    trainer.fit(maskrcnn, datamodule=data_module)

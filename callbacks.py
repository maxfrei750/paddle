import random
from typing import Any

import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from custom_types import Batch
from visualization import visualize_detection

# TODO: Add docstrings


class ExampleDetectionMonitor(Callback):
    def __init__(self) -> None:
        super().__init__()

        self.random_visualization_batch_idx = None

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule):
        self.random_visualization_batch_idx = random.randint(0, sum(trainer.num_val_batches) - 1)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ):
        # Trigger on random batch.
        if batch_idx == self.random_visualization_batch_idx:

            # Get a random image from the batch.
            images, _ = batch
            image = random.sample(images, 1)[0]
            image = image.to(pl_module.device)

            # Get a prediction.
            prediction = pl_module([image])[0]

            # log prediction image
            detection_image = np.array(visualize_detection(image, prediction, score_threshold=0.5))

            trainer.logger.experiment.add_image(
                "validation/example_detection",
                detection_image,
                global_step=trainer.global_step,
                dataformats="HWC",
            )

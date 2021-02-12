import random
from typing import Any

import numpy as np
from pytorch_lightning import LightningModule, Trainer, callbacks

from ..custom_types import Batch
from ..visualization import visualize_detection


class ExampleDetectionMonitor(callbacks.Callback):
    """Callback that creates example detections and logs them to tensorboard. The input images are
    randomly sampled from the validation data set."""

    def __init__(self) -> None:
        super().__init__()

        self.random_visualization_batch_idx = None

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Randomly select a validation batch, from which the random input image for the example
            detection is gonna be sampled from.

        :param trainer: Lightning Trainer
        :param pl_module: Lightning Module
        """
        self.random_visualization_batch_idx = random.randint(0, sum(trainer.num_val_batches) - 1)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """If the current batch was randomly sampled in `on_validation_epoch_start`, then an image
            from the batch is randomly sampled, input into the model and the resulting detection is
            visualized and saved in tensorboard.

        :param trainer: Lightning Trainer
        :param pl_module: Lightning Module
        :param outputs: Gathered outputs from all validation batches.
        :param batch: Current batch.
        :param batch_idx: Current batch id.
        :param dataloader_idx: Dataloader id.
        """
        # Trigger on random batch.
        if batch_idx == self.random_visualization_batch_idx:

            # Get a random image from the batch.
            images, _ = batch
            image = random.sample(images, 1)[0]
            image = image.to(pl_module.device)

            # Get a prediction.
            prediction = pl_module([image])[0]

            # Log prediction image.
            detection_image = np.array(visualize_detection(image, prediction))

            trainer.logger.experiment.add_image(
                "validation/example_detection",
                detection_image,
                global_step=trainer.global_step,
                dataformats="HWC",
            )

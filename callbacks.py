import random

import numpy as np
from pytorch_lightning.callbacks import Callback

from visualization import visualize_detection


class ExampleDetectionMonitor(Callback):
    def __init__(self) -> None:
        super().__init__()

        self.random_visualization_batch_idx = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self.random_visualization_batch_idx = random.randint(0, sum(trainer.num_val_batches) - 1)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
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

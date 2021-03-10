import random
from typing import Any, Optional

import numpy as np
from pytorch_lightning import LightningModule, Trainer, callbacks

from ..custom_types import Batch
from ..postprocessing import filter_low_score_instances
from ..visualization import visualize_annotation


class ExampleDetectionMonitor(callbacks.Callback):
    """Callback that creates example detections and logs them to tensorboard. The input images are
        randomly sampled from the validation data set.

    :param score_threshold: Threshold, below which instances are removed.
    :param do_display_box: If true and available, then bounding boxes are displayed.
    :param do_display_label: If true and available, then labels are displayed.
    :param do_display_score: If true and available, then scores are displayed.
    :param do_display_mask: If true and available, then masks are displayed.
    :param do_display_outlines_only: If true, only the outlines of masks are displayed.
    :param line_width: Line width for bounding boxes and mask outlines.
    :param font_size: Font Size for labels and scores.

    """

    def __init__(
        self,
        score_threshold: float = 0,
        do_display_box: Optional[bool] = True,
        do_display_label: Optional[bool] = True,
        do_display_score: Optional[bool] = True,
        do_display_mask: Optional[bool] = True,
        do_display_outlines_only: Optional[bool] = True,
        line_width: Optional[int] = 3,
        font_size: Optional[int] = 16,
    ) -> None:
        super().__init__()

        self.font_size = font_size
        self.line_width = line_width
        self.do_display_outlines_only = do_display_outlines_only
        self.do_display_mask = do_display_mask
        self.do_display_score = do_display_score
        self.do_display_label = do_display_label
        self.do_display_box = do_display_box
        self.score_threshold = score_threshold
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
        :param outputs: Gathered output from all validation batches.
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

            prediction = filter_low_score_instances(
                prediction, score_threshold=self.score_threshold
            )

            map_label_to_class_name = getattr(
                pl_module.train_dataloader().dataset, "map_label_to_class_name", None
            )

            # Log prediction image.
            detection_image = np.array(
                visualize_annotation(
                    image,
                    prediction,
                    do_display_label=self.do_display_label,
                    do_display_score=self.do_display_score,
                    do_display_outlines_only=self.do_display_outlines_only,
                    do_display_box=self.do_display_box,
                    do_display_mask=self.do_display_mask,
                    line_width=self.line_width,
                    font_size=self.font_size,
                    map_label_to_class_name=map_label_to_class_name,
                )
            )

            trainer.logger.experiment.add_image(
                "validation/example_detection",
                detection_image,
                global_step=trainer.global_step,
                dataformats="HWC",
            )

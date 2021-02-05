import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from pytorch_lightning import LightningModule, Trainer, callbacks
from torchvision.transforms import ToPILImage

from custom_types import AnyPath, Batch, TestOutput
from utilities import dictionary_to_cpu
from visualization import visualize_detection


class ExampleDetectionMonitor(callbacks.Callback):
    """Callback that creates example detections and logs them to tensorboard. The input images are
    randomly sampled from the validation data set."""

    def __init__(self) -> None:
        super().__init__()

        self.random_visualization_batch_idx = None

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Randomly select a validation batch, from which the random input image for the example detection is gonna be
            sampled from.

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
            detection_image = np.array(visualize_detection(image, prediction, score_threshold=0.5))

            trainer.logger.experiment.add_image(
                "validation/example_detection",
                detection_image,
                global_step=trainer.global_step,
                dataformats="HWC",
            )


class ModelCheckpoint(callbacks.ModelCheckpoint):
    """Replaces slashes in the checkpoint filename with underscores to prevent the unwanted
    creation of directories."""

    # TODO: Replace with pytorch_lightning.callbacks.ModelCheckpoint as soon as
    #  https://github.com/PyTorchLightning/pytorch-lightning/issues/4012 is resolved.

    @classmethod
    def _format_checkpoint_name(
        cls,
        filename: Optional[str],
        epoch: int,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> str:
        filename = super()._format_checkpoint_name(filename, epoch, step, metrics, prefix)
        filename = filename.replace("/", "_")
        return filename


class TestPredictionWriter(callbacks.Callback):
    def __init__(self, output_root: AnyPath) -> None:
        """Saves test detections in form of a MaskRCNNDataset.

        :param output_root: Root folder for the output.
        """
        super().__init__()
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: TestOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Save the detections of the current batch in form of a dataset.

        :param trainer: Lightning Trainer
        :param pl_module: Lightning Module
        :param outputs: Outputs of the current test batch.
        :param batch: Current batch.
        :param batch_idx: Current batch id.
        :param dataloader_idx: Dataloader id.
        """
        images, targets = batch

        predictions = outputs["predictions"]

        for image, target, prediction in zip(images, targets, predictions):

            # Fetch data from GPU.
            image = image.cpu()
            target = dictionary_to_cpu(target)
            prediction = dictionary_to_cpu(prediction)

            # Store original image.
            image_name = target["image_name"]
            image_file_path = self.output_root / f"image_{image_name}.png"
            ToPILImage()(image).save(image_file_path)

            # Separate detections based on label.
            labels = prediction["labels"]
            scores = prediction["scores"]
            masks = prediction["masks"]

            unique_labels = prediction["labels"].unique()

            for label in unique_labels:
                # Prepare output folder for current label.
                label_folder_path = self.output_root / f"{label}"
                label_folder_path.mkdir(parents=True, exist_ok=True)

                # Select relevant masks and scores.
                is_current_label = labels == label
                label_masks = masks[is_current_label]
                label_scores = scores[is_current_label]

                # Save masks.
                num_masks = len(label_masks)
                mask_file_names = [
                    f"mask_{image_name}_{mask_idx}.png" for mask_idx in range(num_masks)
                ]

                for mask_file_name, mask in zip(mask_file_names, label_masks):
                    mask_file_path = label_folder_path / mask_file_name
                    mask = (mask >= 0.5).double()
                    ToPILImage()(mask).save(mask_file_path)

                # Save scores.
                score_file_path = label_folder_path / f"scores_{image_name}.csv"
                pd.DataFrame({"score": label_scores.cpu().numpy()}, index=mask_file_names).to_csv(
                    score_file_path
                )

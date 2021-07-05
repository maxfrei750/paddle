from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from pytorch_lightning import LightningModule, Trainer, callbacks
from torchvision.transforms import ToPILImage

from ...custom_types import AnyPath, Batch, TestOutput
from ...data.utilities import dictionary_to_cpu
from ...postprocessing import filter_empty_instances


class PredictionWriter(callbacks.Callback):
    """Saves test detections in form of a MaskRCNNDataset.

    :param output_root: Root folder for the output.
    :param map_label_to_class_name: Dictionary, which maps instance labels to class names.
    """

    def __init__(
        self,
        output_root: AnyPath,
        map_label_to_class_name: Optional[Dict[int, str]] = None,
    ) -> None:
        super().__init__()
        self.map_label_to_class_name = map_label_to_class_name
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

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

            # Filter empty detections.
            prediction = filter_empty_instances(prediction)

            # Separate detections based on label.
            labels = prediction["labels"]
            scores = prediction["scores"]
            masks = prediction["masks"]

            unique_labels = prediction["labels"].unique()

            for label in unique_labels:

                label = int(label)

                # Prepare output folder for current label.
                label_name = (
                    self.map_label_to_class_name[label]
                    if self.map_label_to_class_name is not None
                    and label in self.map_label_to_class_name
                    else str(label)
                )

                label_folder_path = self.output_root / label_name

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
                    if mask_file_path.exists():
                        raise FileExistsError(f"File already exists: {mask_file_path}")
                    ToPILImage()(mask).save(mask_file_path)

                # Save scores.
                score_file_path = label_folder_path / f"scores_{image_name}.csv"
                pd.DataFrame({"score": label_scores.cpu().numpy()}, index=mask_file_names).to_csv(
                    score_file_path
                )

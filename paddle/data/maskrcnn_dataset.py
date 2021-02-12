from itertools import compress
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils
from PIL import Image as PILImage
from torchvision.transforms import functional as F

from paddle.custom_types import Annotation, AnyPath, Image, Mask

from .utilities import extract_bounding_boxes


class MaskRCNNDataset(torch.utils.data.Dataset):
    """Data set for Mask R-CNN training, validation or test data.

    :param root: Path, where data is stored in the following structure:
        root/
            subset/
                image_a.png
                image_b.png
                image_c.png
                ...
                classname1/
                    scores_a....csv
                    mask_a....png
                    mask_a....png
                    mask_a....png
                    ...
                    scores_b....csv
                    mask_b....png
                    mask_b....png
                    ...
                classname2/
                    scores_a....csv
                    mask_a....png
                    mask_a....png
                    mask_a....png
                    ...
                    scores_b....csv
                    mask_b....png
                    mask_b....png
                    ...
            subset2/
                ...
            ...
    :param subset: Name of the subset to use.
    :param transform: torchvision or albumentation transform.
    """

    def __init__(
        self,
        root: AnyPath,
        subset: str,
        transform: Optional[Any] = None,
    ) -> None:

        root = Path(root)
        self.root = root
        assert root.is_dir(), "The specified root does not exist: " + str(root)

        self.subset_path = root / subset
        assert self.subset_path.is_dir(), "The specified subset folder does not exist: " + subset

        self.subset = subset

        self.image_paths = list(self.subset_path.glob("image_*.*"))

        assert self.image_paths, "No images found based on 'image_*.*'."

        self.transform = transform

        self._gather_class_names()

        self.map_label_to_class_name = {
            label: class_name for (label, class_name) in enumerate(self.class_names)
        }
        self.map_class_name_to_label = {
            class_name: label for (label, class_name) in enumerate(self.class_names)
        }

        self.num_classes = len(self.class_names)

    def _gather_class_names(self) -> None:
        """Gather class names based on class folders in subset folder."""
        class_names = [f.name for f in self.subset_path.iterdir() if f.is_dir()]

        if not class_names:
            raise FileNotFoundError(
                f"Cannot create class name dictionary, because there are no class folders in "
                f"{self.subset_path}."
            )

        class_names = ["background"] + class_names
        self.class_names = class_names

    def __getitem__(self, index: int) -> Tuple[Image, Annotation]:
        """Retrieve a sample from the dataset.

        :param index: Index of the sample to retrieve.
        :return: Tuple containing an image as torch tensor and a dict holding the available ground
            truth data.
        """

        image_path = self.image_paths[index]
        image_name = image_path.stem[6:]

        image = PILImage.open(image_path).convert("RGB")

        scores = []
        masks = []
        labels = []

        for class_name in self.class_names:
            class_folder_path = self.subset_path / class_name

            mask_paths = list(class_folder_path.glob(f"mask_{image_name}*.*"))

            if not mask_paths:
                continue

            num_masks = len(mask_paths)

            labels += [self.map_class_name_to_label[class_name]] * num_masks

            score_file_path = class_folder_path / f"scores_{image_name}.csv"

            if score_file_path.exists():
                unsorted_scores = pd.read_csv(score_file_path, index_col=0)
                num_scores = len(unsorted_scores)
                assert num_scores == num_masks, (
                    f"Inconsistent number of masks ({num_masks}) and scores({num_scores}) for "
                    f"{image_path} (class '{class_name}')."
                )

                mask_file_names = [mask_path.name for mask_path in mask_paths]
                scores += list(unsorted_scores["score"][mask_file_names])
            else:
                scores += [1.0] * num_masks

            for mask_path in mask_paths:
                mask = PILImage.open(mask_path).convert("1")

                assert (
                    image.size == mask.size
                ), f"Size of mask {mask_path.parts[-2:]} differs from image size."

                mask = np.array(mask)
                masks.append(mask)

        num_instances = len(masks)

        image = np.array(image)

        if self.transform is not None:
            transformed_data = self.transform(image=image, masks=masks)

            image = transformed_data["image"]
            masks = transformed_data["masks"]

            # Filter empty masks.
            masks, labels = self._remove_empty_masks(masks, labels)

            num_instances = len(masks)

        boxes = extract_bounding_boxes(masks)

        if len(boxes):
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            areas = []

        labels = torch.as_tensor(labels)
        scores = torch.as_tensor(scores)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.array(masks, dtype=np.uint8))
        image_id = torch.tensor([index])
        areas = torch.as_tensor(areas)
        # Assume that there are no crowd instances.
        iscrowd = torch.zeros((num_instances,), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
            "image_name": image_name,
        }

        image = F.to_tensor(image)

        return image, target

    def __len__(self) -> int:
        """Retrieve the number of samples in the data set."""
        return len(self.image_paths)

    @staticmethod
    def _remove_empty_masks(masks: List[Mask], labels: List[int]) -> Tuple[List[Mask], List[int]]:
        """Remove empty masks from list of masks.

        :param masks: List of masks (HxW numpy arrays).
        :return: List of non-empty masks (HxW numpy arrays with at least one non-zero element) and
         list of associated labels.
        """

        is_not_empty = [np.any(mask) for mask in masks]

        masks = list(compress(masks, is_not_empty))
        labels = list(compress(labels, is_not_empty))

        return masks, labels

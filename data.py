import multiprocessing
from itertools import compress
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import albumentations
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.utils.data
from PIL import Image as PILImage
from torchvision.transforms import functional as F

from custom_types import Annotation, AnyPath, Batch, Image, Mask
from utilities import dictionary_to_device


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
                f"Cannot create class name dictionary, because there are no class folders in {self.subset_path}."
            )

        class_names = ["background"] + class_names
        self.class_names = class_names

    def __getitem__(self, index: int) -> Tuple[Image, Annotation]:
        """Retrieve a sample from the dataset.

        :param index: Index of the sample to retrieve.
        :return: Tuple containing an image as torch tensor and a dict holding the available ground truth data.
        """

        image_path = self.image_paths[index]
        image_name = image_path.stem[6:]

        image = PILImage.open(image_path).convert("RGB")
        image = np.array(image)

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
                mask = np.array(mask)
                masks.append(mask)

        num_instances = len(masks)

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


def extract_bounding_boxes(masks: List[Mask]) -> np.ndarray:
    """Extract the bounding boxes of multiple masks.

    :param masks: List of N masks (HxW numpy arrays).
    :return: Nx4 numpy array of bounding boxes.
    """
    boxes = list()
    for mask in masks:
        box = extract_bounding_box(mask)
        boxes.append(box)

    boxes = np.asarray(boxes)

    return boxes


def extract_bounding_box(mask: Mask) -> Tuple[int, int, int, int]:
    """Extract the bounding box of a mask.

    :param mask: HxW numpy array
    :return: bounding box (Tuple[int, int, int, int]).
    """
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1]) + 1
    ymin = np.min(pos[0])
    ymax = np.max(pos[0]) + 1
    return xmin, ymin, xmax, ymax


class MaskRCNNDataModule(pl.LightningDataModule):
    """LightningDataModule to supply Mask R-CNN training, validation and test data.

    :param data_root: Path, where data is stored in the following structure:
        data_root/
            training/
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
            validation/
                ...
            test/
                ...
    :param cropping_rectangle: If not None, [x0, y0, x1, y1] rectangle used for the cropping of
        images. Applied before all other transforms.
    :param batch_size: Number of samples per batch.
    :param num_workers: Number of workers to load data. If None, defaults to the number of threads of the CPU.
    """

    def __init__(
        self,
        data_root: AnyPath,
        cropping_rectangle: Optional[Tuple[int, int, int, int]] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        train_subset: Optional[str] = None,
        val_subset: Optional[str] = None,
        test_subset: Optional[str] = None,
    ) -> None:

        super().__init__()

        self.data_root = Path(data_root)
        self.cropping_rectangle = cropping_rectangle
        self.batch_size = batch_size

        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers

        self.train_subset = train_subset
        self.val_subset = val_subset
        self.test_subset = test_subset

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transforms = self.get_transforms(train=True)
        self.val_transforms = self.get_transforms(train=False)
        self.test_transforms = self.get_transforms(train=False)

        self.map_label_to_class_name = None
        self.num_classes = None

    def prepare_data(self) -> None:
        """Do nothing."""
        pass

    def setup(self, stage: Optional[Literal["fit", "test"]] = None) -> None:
        """Set up the training, validation and test data sets.

        :param stage: Either "fit", when used for training and validation or "test", when used for testing of a model.
        """

        if stage == "fit" or stage is None:
            if self.train_subset is None:
                if stage == "fit":
                    raise ValueError("No train_subset specified.")
            else:
                self.train_dataset = MaskRCNNDataset(
                    self.data_root,
                    subset=self.train_subset,
                    transform=self.train_transforms,
                )

            if self.val_subset is None:
                if stage == "fit":
                    raise ValueError("No val_subset specified.")
            else:
                self.val_dataset = MaskRCNNDataset(
                    self.data_root,
                    subset=self.val_subset,
                    transform=self.val_transforms,
                )

        if stage == "test" or stage is None:
            if self.test_subset is None:
                if stage == "test":
                    raise ValueError("No test_subset specified.")
            else:
                self.test_dataset = MaskRCNNDataset(
                    self.data_root,
                    subset=self.test_subset,
                    transform=self.test_transforms,
                )

    def get_transforms(self, train: bool = False) -> albumentations.Compose:
        """Compose transforms for image preprocessing (e.g. cropping) and augmentation (only for training).

        :param train: Specify, whether to apply image augmentation or not.
        :return: Composed transforms.
        """
        transforms = []

        if self.cropping_rectangle:
            transforms.append(albumentations.Crop(*self.cropping_rectangle))

        if train:
            transforms += [
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.RandomRotate90(always_apply=True),
            ]

        return albumentations.Compose(transforms)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a dataloader for training."""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a dataloader for validation."""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a dataloader for testing."""
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def transfer_batch_to_device(self, batch: Batch, device: torch.device) -> Batch:
        """Transfer a uncollated_batch of data to a device (e.g. GPU).

        :param batch: Batch of data. Tuple containing a tuple of images and a tuple of targets.
        :param device: Target device.
        :return: Batch on target device.
        """
        images, targets = batch

        # TODO: Check if tuples can be replaced with lists.

        images = tuple(image.to(device) for image in images)
        targets = tuple(dictionary_to_device(target, device) for target in targets)

        return images, targets

    @staticmethod
    def collate(uncollated_batch: List[Tuple[Image, Annotation]]) -> Batch:
        """Defines how to collate batches.

        :param uncollated_batch: Uncollated batch of data. List containing tuples of images and targets.
        :return: Collated batch. Tuple containing a tuple of images and a tuple of targets.
        """
        return tuple(zip(*uncollated_batch))

import multiprocessing
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

import albumentations
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import functional as F

from utilities import AnyPath

# Custom types
Mask = np.ndarray
Batch = Tuple[Tuple[torch.Tensor, ...], Tuple[dict, ...]]


class MaskRCNNDataset(torch.utils.data.Dataset):
    """Data set for Mask R-CNN training, validation or test data.

    :param root: Path, where data is stored in the following structure:
        root/
            subset/
                image_a.png
                image_b.png
                image_c.png
                ...
                mask_a_1.png
                mask_a_2.png
                mask_a_3.png
                ...
                mask_b_1.png
                mask_b_2.png
                ...
            subset2/
                ...
            ...
    :param subset: Name of the subset to use.
    :param transform: torchvision or albumentation transform.
    :param class_names: Dictionary, to map class indices to class names.
    """

    def __init__(
        self,
        root: AnyPath,
        subset: str,
        transform: Optional[Any] = None,
        class_names: Optional[dict] = None,
    ) -> None:

        root = Path(root)
        self.root = root
        assert root.is_dir(), "The specified root does not exist: " + root

        self.subset_path = root / subset
        assert self.subset_path.is_dir(), "The specified subset folder does not exist: " + subset

        self.subset = subset

        self.image_paths = list(self.subset_path.glob("image_*.*"))

        assert self.image_paths, "No images found based on 'image_*.*'."

        self.transform = transform

        if class_names is None:
            self.class_name_dict = {1: "particle"}
        else:
            self.class_name_dict = class_names

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        """Retrieve a sample from the dataset.

        :param index: Index of the sample to retrieve.
        :return: Tuple containing an image as torch tensor and a dict holding the available ground truth data.
        """

        image_path = self.image_paths[index]

        image_name = image_path.stem[6:]

        mask_paths = list(self.subset_path.glob(f"mask_{image_name}*.*"))

        # TODO: Support multiple classes.

        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image)

        num_instances = len(mask_paths)

        masks = list()

        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert("1")
            mask = np.array(mask)
            masks.append(mask)

        if self.transform is not None:
            transformed_data = self.transform(image=image, masks=masks)

            image = transformed_data["image"]
            masks = transformed_data["masks"]

            # Filter empty masks.
            masks = self._remove_empty_masks(masks)

            num_instances = len(masks)

        # TODO: Support multiple classes.
        labels = np.ones((num_instances,), dtype=np.int64)
        boxes = extract_bounding_boxes(masks)

        if len(boxes):
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            areas = []

        labels = torch.as_tensor(labels)
        scores = torch.ones((num_instances,), dtype=torch.float32)
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
    def _remove_empty_masks(masks: List[Mask]):
        """Remove empty masks from list of masks.

        :param masks: List of masks (HxW numpy arrays).
        :return: List of non-empty masks (HxW numpy arrays with at least one non-zero element).
        """

        return [mask for mask in masks if np.any(mask)]


def extract_bounding_boxes(masks: List[Mask]):
    """Extract the bounding boxes of multiple masks.

    :param masks: List of masks (HxW numpy arrays).
    :return: List of bounding boxes (4x1 numpy arrays).
    """
    boxes = list()
    for mask in masks:
        box = extract_bounding_box(mask)
        boxes.append(box)

    boxes = np.asarray(boxes)

    return boxes


def extract_bounding_box(mask: Mask):
    """Extract the bounding box of a mask.

    :param mask: HxW numpy array
    :return: bounding box (4x1 numpy array).
    """
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1]) + 1
    ymin = np.min(pos[0])
    ymax = np.max(pos[0]) + 1
    return [xmin, ymin, xmax, ymax]


class MaskRCNNDataModule(pl.LightningDataModule):
    """LightningDataModule to supply Mask R-CNN training, validation and test data.

    :param data_root: Path, where data is stored in the following structure:
        data_root/
            training/
                image_a.png
                image_b.png
                image_c.png
                ...
                mask_a_1.png
                mask_a_2.png
                mask_a_3.png
                ...
                mask_b_1.png
                mask_b_2.png
                ...
            validation/
                ...
            test/
                ...
    :param class_names: Dictionary, to map class indices to class names.
    :param cropping_rectangle: If not None, [x0, y0, x1, y1] rectangle used for the cropping of
        images. Applied before all other transforms.
    :param batch_size: Number of samples per batch.
    :param num_workers: Number of workers to load data. If None, defaults to the number of threads of the CPU.
    """

    def __init__(
        self,
        data_root: AnyPath,
        class_names: Optional[dict] = None,
        cropping_rectangle: Optional[Tuple[int, int, int, int]] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        train_subset="training",
        val_subset="validation",
        test_subset="test",
    ) -> None:

        super().__init__()

        self.data_root = Path(data_root)
        self.class_name_dict = class_names
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

    def prepare_data(self) -> None:
        """Do nothing."""
        pass

    def setup(self, stage: Optional[Literal["fit", "test"]] = None) -> None:
        """Set up the training, validation and test data sets.

        :param stage: Either "fit", when used for training and validation or "test", when used for testing of a model.
        """

        if stage == "fit" or stage is None:
            self.train_dataset = MaskRCNNDataset(
                self.data_root,
                subset=self.train_subset,
                transform=self.train_transforms,
                class_names=self.class_name_dict,
            )

            self.val_dataset = MaskRCNNDataset(
                self.data_root,
                subset=self.val_subset,
                transform=self.val_transforms,
                class_names=self.class_name_dict,
            )

        if stage == "test" or stage is None:
            self.test_dataset = MaskRCNNDataset(
                self.data_root,
                subset=self.test_subset,
                transform=self.test_transforms,
                class_names=self.class_name_dict,
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

        images = tuple(image.to(device) for image in images)
        targets = tuple(
            {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets
        )

        return images, targets

    @staticmethod
    def collate(uncollated_batch: List[Tuple[torch.Tensor, dict]]) -> Batch:
        """Defines how to collate batches.

        :param uncollated_batch: Uncollated batch of data. List containing tuples of images and targets.
        :return: Collated batch. Tuple containing a tuple of images and a tuple of targets.
        """
        return tuple(zip(*uncollated_batch))

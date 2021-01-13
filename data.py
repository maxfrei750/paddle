import multiprocessing
from pathlib import Path
from typing import Any, Optional, Tuple

import albumentations
import numpy as np
import pytorch_lightning as pl
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from utilities import AnyPath


class MaskRCNNDataset(torch.utils.data.Dataset):
    def __init__(self, root, subset, transforms=None, class_name_dict=None):

        root = Path(root)
        self.root = root
        assert root.is_dir(), "The specified root does not exist: " + root

        self.subset_path = root / subset
        assert self.subset_path.is_dir(), "The specified subset folder does not exist: " + subset

        self.subset = subset

        # For large datasets it might be better to keep the glob-generator.
        self.image_paths = list(self.subset_path.glob("image_*.*"))

        assert self.image_paths, "No images found based on 'image_*.*'."

        self.transforms = transforms

        if class_name_dict is None:
            self.class_name_dict = {1: "particle"}
        else:
            self.class_name_dict = class_name_dict

    def __getitem__(self, index):
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

        if self.transforms is not None:
            transformed_data = self.transforms(image=image, masks=masks)

            image = transformed_data["image"]
            masks = transformed_data["masks"]

            # Filter empty masks.
            masks = _remove_empty_masks(masks)

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

    def __len__(self):
        return len(self.image_paths)


def extract_bounding_boxes(masks):
    boxes = list()
    for mask in masks:
        box = extract_bounding_box(mask)
        boxes.append(box)

    boxes = np.asarray(boxes)

    return boxes


def extract_bounding_box(mask):
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1]) + 1
    ymin = np.min(pos[0])
    ymax = np.max(pos[0]) + 1
    box = [xmin, ymin, xmax, ymax]
    return box


def _remove_empty_masks(masks):
    return [mask for mask in masks if np.any(mask)]


class MaskRCNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root: AnyPath,
        class_name_dict: Optional[dict] = None,
        cropping_rectangle: Optional[Tuple[int, int, int, int]] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
    ):
        super().__init__()

        self.data_root = Path(data_root)
        self.class_name_dict = class_name_dict
        self.cropping_rectangle = cropping_rectangle
        self.batch_size = batch_size

        if num_workers is None:
            self.num_workers = multiprocessing.cpu_count()
        else:
            self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.train_transforms = self.get_transforms(train=True)
        self.val_transforms = self.get_transforms(train=False)
        self.test_transforms = self.get_transforms(train=False)

    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MaskRCNNDataset(
                self.data_root,
                subset="training",
                transforms=self.train_transforms,
                class_name_dict=self.class_name_dict,
            )

            self.val_dataset = MaskRCNNDataset(
                self.data_root,
                subset="validation",
                transforms=self.val_transforms,
                class_name_dict=self.class_name_dict,
            )

            # self.dims = tuple(self.train_dataset[0][0].shape)

        if stage == "test" or stage is None:
            self.test_dataset = MaskRCNNDataset(
                self.data_root,
                subset="test",
                transforms=self.test_transforms,
                class_name_dict=self.class_name_dict,
            )

            # self.dims = tuple(self.test_dataset[0][0].shape)

    def get_transforms(self, train=False) -> albumentations.Compose:
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

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        images, targets = batch

        images = tuple(image.to(device) for image in images)
        targets = tuple(
            {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets
        )

        return images, targets

    @staticmethod
    def collate(batch):
        return tuple(zip(*batch))

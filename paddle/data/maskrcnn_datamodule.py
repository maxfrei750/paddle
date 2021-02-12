from pathlib import Path
from typing import Literal, Optional, Tuple

import albumentations
import pytorch_lightning as pl
import torch.utils

from paddle.custom_types import AnyPath, Batch

from .maskrcnn_dataloader import MaskRCNNDataLoader
from .maskrcnn_dataset import MaskRCNNDataset
from .utilities import dictionary_to_device


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
    """

    def __init__(
        self,
        data_root: AnyPath,
        cropping_rectangle: Optional[Tuple[int, int, int, int]] = None,
        batch_size: int = 1,
        train_subset: Optional[str] = None,
        val_subset: Optional[str] = None,
        test_subset: Optional[str] = None,
    ) -> None:

        super().__init__()

        self.data_root = Path(data_root)
        self.cropping_rectangle = cropping_rectangle
        self.batch_size = batch_size

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

        :param stage: Either "fit", when used for training and validation or "test", when used for
            testing of a model.
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
        """Compose transforms for image preprocessing (e.g. cropping) and augmentation (only for
            training).

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
        return MaskRCNNDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a dataloader for validation."""
        return MaskRCNNDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns a dataloader for testing."""
        return MaskRCNNDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
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

import warnings
from itertools import compress
from typing import Any

import albumentations
import numpy as np
import pandas as pd
import torch
from PIL import Image as PILImage
from torchvision.transforms import functional as F

from .utilities import extract_bounding_box, image_size_hwc


class Sample:
    def __init__(self, parent_dataset: Any, index: int) -> None:
        """Single sample of a dataset, i.e. image plus target data.

        :param parent_dataset: dataset the sample belongs to
        :param index: index of the sample in the dataset (not necessarily the same as the image
            index, since images can be spliced into multiple parts.
        """

        self.parent = parent_dataset
        self.index = index

        self.transforms = []

        self.scores = []
        self.masks = []
        self.labels = []
        self.boxes = []
        self.areas = []

        self._read_image()
        self._gather_raw_target_data()
        self._configure_transforms()
        self._apply_transforms()

        self._extract_bounding_boxes_from_masks()
        self._calculate_bounding_box_areas()

    def _configure_transforms(self):
        """Configure initial cropping, slicing and user transforms."""
        self._configure_initial_cropping()
        self._configure_slicing()
        self._configure_user_transform()

    def _apply_transforms(self):
        """Apply transforms to image and target data."""

        if self.transforms:
            transform = albumentations.Compose(self.transforms)
            transformed_data = transform(image=self.image, masks=self.masks)

            self.image = transformed_data["image"]
            self.masks = transformed_data["masks"]

        self._filter_empty_masks()

    def _gather_raw_target_data(self):
        """Gather target data (masks, labels and scores (if present)."""
        for class_name in self.parent.class_names:
            self._gather_masks_for_class(class_name)
            self._create_labels_for_class(class_name)
            self._gather_scores_for_class(class_name)

        if self.num_instances == 0:
            raise FileNotFoundError(f"Sample '{self.image_name}' features no instances.")

    def _gather_scores_for_class(self, class_name: str):
        """Gather scores for a single class, if present.

        :param class_name: name of the class
        """

        mask_paths_class = self._gather_mask_paths_for_class(class_name)

        if not mask_paths_class:
            return

        num_instances_class = len(mask_paths_class)
        score_file_path = (
            self.parent.subset_path
            / class_name
            / f"{self.parent.scores_prefix}{self.image_name}.csv"
        )
        if score_file_path.exists():
            unsorted_scores = pd.read_csv(score_file_path, index_col=0)
            num_scores = len(unsorted_scores)
            assert num_scores == num_instances_class, (
                f"Inconsistent number of masks ({num_instances_class}) and scores({num_scores}) "
                f"for {self.image_path} (class '{class_name}')."
            )

            mask_file_names = [mask_path.name for mask_path in mask_paths_class]
            self.scores += list(unsorted_scores["score"][mask_file_names])
        else:
            self.scores += [1.0] * num_instances_class

    def _create_labels_for_class(self, class_name):
        """Create labels for the instances of a class.

        :param class_name: name of the class
        """
        mask_paths_class = self._gather_mask_paths_for_class(class_name)

        if not mask_paths_class:
            return

        num_instances_class = len(mask_paths_class)
        self.labels += [self.parent.map_class_name_to_label[class_name]] * num_instances_class

    def _gather_masks_for_class(self, class_name):
        """Gather masks for a single class.

        :param class_name: name of the class
        """
        mask_paths_class = self._gather_mask_paths_for_class(class_name)

        if not mask_paths_class:
            return

        for mask_path in mask_paths_class:
            mask = PILImage.open(mask_path).convert("1")
            mask = np.array(mask)

            assert image_size_hwc(self.image) == image_size_hwc(
                mask
            ), f"Size of mask {mask_path.parts[-2:]} differs from image size."

            self.masks.append(mask)

    def _gather_mask_paths_for_class(self, class_name: str):
        """Gather paths of mask files for a single class.

        :param class_name: name of the class
        """
        class_folder_path = self.parent.subset_path / class_name
        mask_paths_class = list(
            class_folder_path.glob(f"{self.parent.mask_prefix}{self.image_name}_*.*")
        )
        return mask_paths_class

    def _configure_initial_cropping(self):
        """Configure the transform for the initial image cropping."""
        if self.parent.initial_cropping_rectangle is not None:
            self.transforms.append(albumentations.Crop(*self.parent.initial_cropping_rectangle))

    def _configure_slicing(self):
        """Configure the transformation for the slicing of images."""
        slice_index = self.index % self.parent.num_slices_per_image
        slice_index_x = slice_index // self.parent.num_slices_per_axis
        slice_index_y = slice_index % self.parent.num_slices_per_axis
        if self.parent.num_slices_per_image > 1:

            if self.parent.initial_cropping_rectangle is None:
                image_size = image_size_hwc(self.image)
            else:
                r = self.parent.initial_cropping_rectangle
                image_size = (r[3] - r[1], r[2] - r[0])

            num_surplus_pixels_y, num_surplus_pixels_x = [
                num_pixels % self.parent.num_slices_per_axis for num_pixels in image_size
            ]

            if num_surplus_pixels_y or num_surplus_pixels_x:
                warnings.warn(
                    f"Cannot slice image evenly. Discarding pixels "
                    f"(x: {num_surplus_pixels_x}; y: {num_surplus_pixels_y})."
                )

            slice_size_y, slice_size_x = [
                num_pixels // self.parent.num_slices_per_axis for num_pixels in image_size
            ]

            x_min = slice_index_x * slice_size_x
            x_max = x_min + slice_size_x
            y_min = slice_index_y * slice_size_y
            y_max = y_min + slice_size_y

            slice_rectangle = [x_min, y_min, x_max, y_max]

            self.transforms.append(albumentations.Crop(*slice_rectangle))

            self.image_name += f"_slice{slice_index}"

        self.slice_index = slice_index
        self.slice_index_x = slice_index_x
        self.slice_index_y = slice_index_y

    def _read_image(self):
        """Read the image of the sample and store some additional information."""
        self.image_index = self.index // self.parent.num_slices_per_image
        self.image_path = self.parent.image_paths[self.image_index]
        self.image_name = self.image_path.stem[6:]
        self.image = np.array(PILImage.open(self.image_path).convert("RGB"))

    def _configure_user_transform(self):
        """Configure the transform supplied by the user."""
        if self.parent.user_transform is not None:
            self.transforms.append(self.parent.user_transform)

    def _filter_empty_masks(self):
        """Remove empty masks."""

        # TODO: Parallelize mask removal with dask.
        is_not_empty = [np.any(mask) for mask in self.masks]

        self.masks = list(compress(self.masks, is_not_empty))
        self.labels = list(compress(self.labels, is_not_empty))
        self.scores = list(compress(self.scores, is_not_empty))

    def _extract_bounding_boxes_from_masks(self):
        """Extract bounding boxes from masks."""
        # TODO: Parallelize bounding box extraction with dask.
        self.boxes = [extract_bounding_box(mask) for mask in self.masks]

    def _calculate_bounding_box_areas(self):
        """Calculate the areas of the bounding boxes"""
        boxes = np.asarray(self.boxes)

        if len(boxes):
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            areas = areas.tolist()
        else:
            areas = []

        self.areas = areas

    @property
    def num_instances(self):
        """Determine the number of instances based on the number of masks."""
        return len(self.masks)

    @property
    def iscrowd(self):
        """Set the iscrowd property of every instance to false."""
        # Assume that there are no crowd instances.
        return [False] * self.num_instances

    @property
    def targets_as_tensors(self):
        """Construct a dictionary of torch tensors."""
        return {
            "boxes": torch.as_tensor(self.boxes, dtype=torch.float32),
            "labels": torch.as_tensor(self.labels),
            "scores": torch.as_tensor(self.scores),
            # It's very important that the type conversion of the masks is done by numpy and not
            # torch, because:
            # torch.as_tensor(np.array(self.masks), dtype=torch.uint8) = 255
            # torch.as_tensor(np.array(self.masks, dtype=np.uint8)) = 1
            "masks": torch.as_tensor(np.array(self.masks, dtype=np.uint8)),
            "image_id": torch.as_tensor([self.index]),
            "area": torch.as_tensor(self.areas),
            "iscrowd": torch.as_tensor(self.iscrowd),
            "image_name": self.image_name,
            "slice_index_x": torch.as_tensor(self.slice_index_x),
            "slice_index_y": torch.as_tensor(self.slice_index_y),
        }

    @property
    def image_as_tensor(self):
        """Convert the image to a torch tensor."""
        return F.to_tensor(self.image)

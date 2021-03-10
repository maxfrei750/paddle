import warnings
from itertools import compress
from pathlib import Path
from typing import Any, List, Optional, Tuple

import albumentations
import numpy as np
import pandas as pd
import torch
import torch.utils
from PIL import Image as PILImage
from torchvision.transforms import functional as F

from paddle.custom_types import Annotation, AnyPath, CroppingRectangle, Image, Mask

from .utilities import extract_bounding_boxes

# TODO: Avoid assert and use more specific errors.


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
    :param initial_cropping_rectangle: If not None, [x_min, y_min, x_max, y_max] rectangle used for
        the cropping of images.
    :param num_slices_per_axis: Integer number of slices per image axis. `num_slices_per_axis`=n
        will result in n² pieces. Slicing is performed after the initial cropping and before the
        user transform.
    :param user_transform: torchvision or albumentation transform.
    """

    def __init__(
        self,
        root: AnyPath,
        subset: Optional[str] = None,
        initial_cropping_rectangle: Optional[CroppingRectangle] = None,
        num_slices_per_axis: Optional[int] = 1,
        user_transform: Optional[Any] = None,
    ) -> None:

        root = Path(root)
        self.root = root
        assert root.is_dir(), "The specified root does not exist: " + str(root)

        self.subset_path = root

        if subset is not None:
            self.subset_path /= subset
            assert self.subset_path.is_dir(), (
                "The specified subset folder does not exist: " + subset
            )

        self.subset = subset

        self.image_paths = list(self.subset_path.glob("image_*.*"))

        assert self.image_paths, "No images found in {self.subset_path} based on 'image_*.*'."

        self.num_images = len(self.image_paths)

        self.initial_cropping_rectangle = initial_cropping_rectangle

        if num_slices_per_axis < 1:
            raise ValueError("`num_slices_per_axis` must be >= 1.")

        if not float(num_slices_per_axis).is_integer():
            raise ValueError("`num_slices_per_axis` must be an integer.")

        self.num_slices_per_axis = num_slices_per_axis
        self.num_slices_per_image = num_slices_per_axis ** 2

        self.user_transform = user_transform

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

        if not class_names:  # use a generic class name, if no class folders exist
            class_names = ["particle"]

        class_names = ["background"] + class_names
        self.class_names = class_names

    def __getitem__(self, index: int) -> Tuple[Image, Annotation]:
        """Retrieve a sample from the dataset.

        :param index: Index of the sample to retrieve.
        :return: Tuple containing an image as torch tensor and a dict holding the available ground
            truth data.
        """

        transforms = []

        if self.initial_cropping_rectangle is not None:
            transforms.append(albumentations.Crop(*self.initial_cropping_rectangle))

        image_index = index // self.num_slices_per_image
        slice_index = index % self.num_slices_per_image

        image_path = self.image_paths[image_index]
        image_name = image_path.stem[6:]

        image = PILImage.open(image_path).convert("RGB")

        slice_index_x = slice_index // self.num_slices_per_axis
        slice_index_y = slice_index % self.num_slices_per_axis

        if self.num_slices_per_image > 1:

            image_name += f"_slice{slice_index}"

            if self.initial_cropping_rectangle is None:
                image_size = image.size
            else:
                r = self.initial_cropping_rectangle
                image_size = (r[2] - r[0], r[3] - r[1])

            num_surplus_pixels_x, num_surplus_pixels_y = [
                num_pixels % self.num_slices_per_axis for num_pixels in image_size
            ]

            if num_surplus_pixels_x or num_surplus_pixels_y:
                warnings.warn(
                    f"Cannot slice image evenly. Discarding pixels "
                    f"(x: {num_surplus_pixels_x}; y: {num_surplus_pixels_y})."
                )

            slice_size = [num_pixels // self.num_slices_per_axis for num_pixels in image_size]

            x_min = slice_index_x * slice_size[0]
            x_max = x_min + slice_size[0]
            y_min = slice_index_y * slice_size[1]
            y_max = y_min + slice_size[1]

            slice_rectangle = [x_min, y_min, x_max, y_max]

            transforms.append(albumentations.Crop(*slice_rectangle))

        scores = []
        masks = []
        labels = []

        for class_name in self.class_names:
            class_folder_path = self.subset_path / class_name

            mask_paths = list(class_folder_path.glob(f"mask_{image_name}_*.*"))

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
                mask = PILImage.open(mask_path).convert("L")

                assert (
                    image.size == mask.size
                ), f"Size of mask {mask_path.parts[-2:]} differs from image size."

                mask = np.array(mask)

                masks.append(mask)

        image = np.array(image)

        if self.user_transform is not None:
            transforms.append(self.user_transform)

        if transforms:
            transform = albumentations.Compose(transforms)
            transformed_data = transform(image=image, masks=masks)

            image = transformed_data["image"]
            masks = transformed_data["masks"]

        # Filter masks smaller than 100 pixels (and associated labels and scores).
        masks, labels, scores = self._remove_small_masks(masks, labels, scores, size_threshold=25)

        num_instances = len(masks)

        masks = [mask.astype(bool) for mask in masks]

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
        slice_index_x = torch.as_tensor(slice_index_x)
        slice_index_y = torch.as_tensor(slice_index_y)

        target = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
            "image_name": image_name,
            "slice_index_x": slice_index_x,
            "slice_index_y": slice_index_y,
        }

        image = F.to_tensor(image)

        return image, target

    def __len__(self) -> int:
        """Retrieve the number of samples in the data set."""
        return self.num_images * self.num_slices_per_image

    @staticmethod
    def _remove_small_masks(
        masks: List[Mask], labels: List[int], scores: List[float], size_threshold: int
    ) -> Tuple[List[Mask], List[int], List[float]]:
        """Remove empty masks from list of masks. Also remove associated scores and labels.

        :param masks: List of masks (HxW numpy arrays).
        :param labels: List of labels.
        :param scores: List of scores.
        :param size_threshold: threshold for the size filtering
        :return: List of masks (HxW numpy arrays) with at least `size_threshold` non-zero pixels and
            list of associated labels and scores.
        """

        # TODO: Parallelize mask removal.
        is_large_enough = [np.sum(mask) >= size_threshold for mask in masks]

        masks = list(compress(masks, is_large_enough))
        labels = list(compress(labels, is_large_enough))
        scores = list(compress(scores, is_large_enough))

        return masks, labels, scores

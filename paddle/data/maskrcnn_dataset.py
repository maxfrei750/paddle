from pathlib import Path
from typing import Any, Optional, Tuple

import torch.utils

from ..custom_types import Annotation, AnyPath, CroppingRectangle, Image
from .sample import Sample


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

        self.mask_prefix = "mask_"
        self.image_prefix = "image_"
        self.scores_prefix = "scores_"

        root = Path(root)
        self.root = root
        if not root.is_dir():
            raise NotADirectoryError(f"The specified root does not exist: {root}")

        self.subset_path = root

        if subset is not None:
            self.subset_path /= subset
            if not self.subset_path.is_dir():
                raise NotADirectoryError(f"The specified subset folder does not exist: {subset}")

        self.subset = subset

        self.image_paths = list(self.subset_path.glob(f"{self.image_prefix}*.*"))

        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {self.subset_path} based on '{self.image_prefix}*.*'."
            )

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

        sample = Sample(self, index)
        return sample.image_as_tensor, sample.targets_as_tensors

    def __len__(self) -> int:
        """Retrieve the number of samples in the data set."""
        return self.num_images * self.num_slices_per_image

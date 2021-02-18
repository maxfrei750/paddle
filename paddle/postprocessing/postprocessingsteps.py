from pathlib import Path, PosixPath
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from torch import Tensor

from ..custom_types import Annotation, AnyPath, Image
from ..visualization import visualize_annotation
from .functional import filter_border_instances, filter_class_instances, filter_low_score_instances


class PostProcessingStepBase:
    """Base class for postprocessing steps."""

    def __call__(self, image: Image, annotation: Annotation) -> Tuple[Image, Annotation]:
        """This method _must_ always return the input image and the annotation, so that multiple
        post processing steps can be chained. However, post processing steps may modify images and
        annotations (e.g. remove instances).

        :param image: Input image
        :param annotation: Dictionary containing annotation of an image (e.g. masks, bounding boxes,
            etc.)
        :return: Image, and annotation but potentially with altered properties (e.g. removed
            instances).
        """
        return image, annotation

    def __str__(self) -> str:
        """The string representation of the class is a summary of its attributes in yaml format.
        Protected attributes (i.e. attributes that start with an underscore) are omitted."""
        return yaml.dump(
            {
                self.__class__.__name__: {
                    key: str(value) if isinstance(value, PosixPath) else value
                    for key, value in self.__dict__.items()
                    if not key.startswith("_")
                }
            }
        )


class SaveMaskProperties(PostProcessingStepBase):
    """Extract mask properties (using user supplied functions) and save them into a file, along with
    the corresponding labels, scores and image names.

    :param output_file_path: Path of the output CSV file, in which measurements are stored.
    :param measurement_fcns: Dictionary, where keys are measurand names and values are callables,
        that accept a numpy array containing multiple masks.
    """

    def __init__(self, output_file_path: AnyPath, measurement_fcns: Dict[str, Callable]) -> None:
        self.output_file_path = Path(output_file_path)

        self._measurement_fcns = measurement_fcns
        self.measurement_names = list(measurement_fcns.keys())

    def __call__(self, image: Image, annotation: Annotation) -> Tuple[Image, Annotation]:
        """Measure and store mask properties of a single annotation.

        :param image: input image
        :param annotation: Dictionary containing annotation of an image (e.g. masks,
            bounding boxes, etc.)
        :return: image and annotation, both identical to the inputs
        """
        masks = annotation["masks"]
        results = {}

        for measurement_name, measurement_fcn in zip(
            self.measurement_names, self._measurement_fcns.values()
        ):
            results[measurement_name] = measurement_fcn(masks)

        results["score"] = annotation["scores"]
        results["label"] = annotation["labels"]
        results["image_name"] = annotation["image_name"]

        # Write header only, if the file does not yet exist.
        with open(self.output_file_path, "a") as file:
            pd.DataFrame(results).to_csv(file, mode="a", index=False, header=not file.tell())

        return image, annotation


class FilterClasses(PostProcessingStepBase):
    """Remove instances that are not in the `class_labels_to_keep` list.

    :param class_labels_to_keep: List of class labels that are to be kept.
    """

    def __init__(self, class_labels_to_keep: List[int]) -> None:
        self.class_labels_to_keep = class_labels_to_keep

    def __call__(self, image: Image, annotation: Annotation) -> Tuple[Image, Annotation]:
        """Remove instances that are not in `self.class_labels_to_keep`.

        :param image: input image
        :param annotation: Dictionary containing annotation of an image (e.g. masks,
            bounding boxes, etc.)
        :return: original input image and annotation with only instances in
            self.class_labels_to_keep.
        """
        annotation = filter_class_instances(
            annotation, class_labels_to_keep=self.class_labels_to_keep
        )
        return image, annotation


class FilterBorderInstances(PostProcessingStepBase):
    """Remove instances that are closer to the image border than a threshold value.

    :param border_width: Specifies how close instances may be to the image border, before they are
        being removed.
    """

    def __init__(self, border_width: int = 3) -> None:
        self.border_width = border_width

    def __call__(self, image: Image, annotation: Annotation) -> Tuple[Image, Annotation]:
        """Remove border instances from a single annotation.

        :param image: input image
        :param annotation: Dictionary containing annotation of an image (e.g. masks,
            bounding boxes, etc.)
        :return: original input image and annotation without border instances
        """
        annotation = filter_border_instances(annotation, border_width=self.border_width)
        return image, annotation


class FilterScore(PostProcessingStepBase):
    """Remove instances with a score below a certain threshold.

    :param score_threshold: instances with a score below this threshold are removed
    """

    def __init__(self, score_threshold: float) -> None:
        self.score_threshold = score_threshold

    def __call__(self, image: Image, annotation: Annotation) -> Tuple[Image, Annotation]:
        """Remove instances below a certain score threshold from a single annotation.

        :param image: input image
        :param annotation: Dictionary containing annotation of an image (e.g. masks,
            bounding boxes, etc.)
        :return: original input image and annotation without instances with a score below the score
            threshold
        """
        annotation = filter_low_score_instances(annotation, self.score_threshold)
        return image, annotation


class Numpify(PostProcessingStepBase):
    """Converts the input image and tensor values of the annotation dictionary to numpy arrays."""

    def __call__(self, image: Image, annotation: Annotation) -> Tuple[Image, Annotation]:
        """Convert an input image and the tensor values of an annotation dictionary to numpy arrays.
        Non-tensor values are untouched.

        :param image: input image (potentially as a  pytorch tensor)
        :param annotation: Dictionary containing annotation of an image (e.g. masks,
            bounding boxes, etc., potentially as pytorch tensors).
        :return: input image as numpy array and annotation consisting of numpy arrays.
        """
        if isinstance(image, Tensor):
            image = image.cpu().numpy()

        for key, value in annotation.items():
            if isinstance(value, Tensor):
                value = value.cpu().numpy()
                annotation[key] = value

        return image, annotation


class SaveVisualization(PostProcessingStepBase):
    """Overlay an image with an annotation and store it in a folder.

    :param output_root: Path, where visualizations are stored.
    :param file_name_prefix: Prefix for visualization output files.
    :param do_display_box: If true and available, then bounding boxes are displayed.
    :param do_display_label: If true and available, then labels are displayed.
    :param do_display_score: If true and available, then scores are displayed.
    :param do_display_mask: If true and available, then masks are displayed.
    :param do_display_outlines_only: If true, only the outlines of masks are displayed.
    :param map_label_to_class_name: Dictionary, which maps instance labels to class names.
    :param line_width: Line width for bounding boxes and mask outlines.
    :param font_size: Font Size for labels and scores.
    """

    def __init__(
        self,
        output_root: AnyPath,
        file_name_prefix: str = "visualization",
        do_display_box: Optional[bool] = True,
        do_display_label: Optional[bool] = True,
        do_display_score: Optional[bool] = True,
        do_display_mask: Optional[bool] = True,
        do_display_outlines_only: Optional[bool] = True,
        map_label_to_class_name: Optional[Dict[int, str]] = None,
        line_width: Optional[int] = 3,
        font_size: Optional[int] = 16,
    ) -> None:

        self.output_root = Path(output_root)
        self.file_name_prefix = file_name_prefix
        self.do_display_box = do_display_box
        self.do_display_label = do_display_label
        self.do_display_score = do_display_score
        self.do_display_mask = do_display_mask
        self.do_display_outlines_only = do_display_outlines_only
        self.map_label_to_class_name = map_label_to_class_name
        self.line_width = line_width
        self.font_size = font_size

    def __call__(self, image: Image, annotation: Annotation) -> Tuple[Image, Annotation]:
        """Create a visualization of an image and the corresponding annotation and save it.

        :param image: input image
        :param annotation: Dictionary containing annotation of an image (e.g. masks,
            bounding boxes, etc., potentially as pytorch tensors).
        :return: original input image and original annotation
        """
        image_name = annotation["image_name"]

        result = visualize_annotation(
            image,
            annotation,
            do_display_box=self.do_display_box,
            do_display_label=self.do_display_label,
            do_display_score=self.do_display_score,
            do_display_mask=self.do_display_mask,
            do_display_outlines_only=self.do_display_outlines_only,
            map_label_to_class_name=self.map_label_to_class_name,
            line_width=self.line_width,
            font_size=self.font_size,
        )

        visualization_file_path = self.output_root / f"{self.file_name_prefix}_{image_name}.png"
        result.save(visualization_file_path)

        return image, annotation

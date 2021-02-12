import random
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont, ImageOps
from scipy.ndimage.morphology import binary_erosion
from skimage import img_as_float, img_as_ubyte
from torchvision import transforms

from .custom_types import Annotation, ColorFloat, ColorInt, Image
from .data import extract_bounding_box
from .statistics import gmean, gstd


def get_viridis_colors(num_colors: int) -> List[ColorFloat]:
    """Get a number of colors from the viridis color map, but leave out the very saturated colors
        at the beginning and end of the colormap.

    :param num_colors: Number of colors to be retrieved.
    :return: List of colors in float format.
    """
    color_min = (0.231, 0.322, 0.545)
    color_max = (0.369, 0.788, 0.384)

    if num_colors == 1:
        return [color_min]

    colormap = LinearSegmentedColormap.from_list("custom_viridis", colors=[color_min, color_max])

    colors = []

    for i in range(num_colors):
        color = colormap(i / (num_colors - 1))[:3]
        colors.append(color)

    return colors


def get_random_viridis_colors(num_colors: int) -> List[ColorFloat]:
    """Get a number of random colors from the viridis color map.

    :param num_colors: Number of colors to be retrieved.
    :return: List of colors in float format.
    """
    colors = []

    for i_color in range(num_colors):
        # color = cm.hsv(random.uniform(0, 0.6))

        colormap = cm.get_cmap("viridis")
        color = colormap(random.uniform(0, 1))
        colors.append(color)

    return colors


def visualize_detection(
    image: Image,
    annotation: Annotation,
    do_display_box: Optional[bool] = True,
    do_display_label: Optional[bool] = True,
    do_display_score: Optional[bool] = True,
    do_display_mask: Optional[bool] = True,
    do_display_outlines_only: Optional[bool] = True,
    map_label_to_class_name: Optional[Dict[int, str]] = None,
    line_width: Optional[int] = 3,
    font_size: Optional[int] = 16,
) -> PILImage:
    """Overlay an image with an annotation of multiple instances.

    :param image: Image
    :param annotation: Dictionary storing annotation properties (e.g. masks, scores, labels,
        bounding boxes, etc.)
    :param do_display_box: If true and available, then bounding boxes are displayed.
    :param do_display_label: If true and available, then labels are displayed.
    :param do_display_score: If true and available, then scores are displayed.
    :param do_display_mask: If true and available, then masks are displayed.
    :param do_display_outlines_only: If true, only the outlines of masks are displayed.
    :param map_label_to_class_name: Dictionary, which maps instance labels to class names.
    :param line_width: Line width for bounding boxes and mask outlines.
    :param font_size: Font Size for labels and scores.
    :return: A PIL image object of the original image with overlayed annotations.
    """

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.truetype("arial.ttf", font_size)

    assert isinstance(image, torch.Tensor) or isinstance(
        image, np.ndarray
    ), "Expected image to be of class torch.Tensor or numpy.ndarray."

    image = image.squeeze()

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Convert from CWH to HWC if necessary.
    if not image.shape[2] == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, 2)

    image = img_as_ubyte(image)

    image = transforms.ToPILImage()(image)

    for key, value in annotation.items():
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
            annotation[key] = value

    for key in ["masks", "boxes"]:
        if key in annotation:
            num_instances = len(annotation[key])
            break

        raise ValueError("Detection must have either masks or boxes.")

    none_list = [None] * num_instances
    masks = annotation.get("masks", none_list)
    boxes = annotation.get("boxes", none_list)
    scores = annotation.get("scores", none_list)
    labels = annotation.get("labels", none_list)

    result = image.convert("RGB")
    colors_float = get_random_viridis_colors(num_instances)

    for mask, box, color_float, score, label in zip(masks, boxes, colors_float, scores, labels):

        color_int = _color_float_to_int(color_float)

        if mask is not None and do_display_mask:
            mask = mask.squeeze()

            if mask.dtype == "uint8" and mask.max() <= 1:
                mask = img_as_float(mask * 255)
            mask = mask >= 0.5

            result = _overlay_image_with_mask(
                result,
                mask,
                color_int,
                do_display_outlines_only=do_display_outlines_only,
                outline_width=line_width,
            )

        if (do_display_label or do_display_score or do_display_box) and box is None:
            box = extract_bounding_box(mask)

        if box is not None and do_display_box:
            ImageDraw.Draw(result).rectangle(box, outline=color_int, width=line_width)

        caption = None

        if label is not None and do_display_label:
            if map_label_to_class_name is None:
                caption = f"class{label}"
            else:
                caption = map_label_to_class_name[label]

        if score is not None and do_display_score:
            if not do_display_label:
                caption = "Score"

            caption += ": {:.3f}".format(score)

        if label is not None and do_display_label or score is not None and do_display_score:
            x, y = box[:2]
            y -= font_size + 2
            ImageDraw.Draw(result).text((x, y), caption, font=font, fill=color_int)

    return result


def _color_float_to_int(color_float: ColorFloat) -> ColorInt:
    """Convert a color in float format into int format.

    :param color_float: Color in float format.
    :return: Color in int format.
    """
    color_int = (
        int(round(color_float[0] * 255)),
        int(round(color_float[1] * 255)),
        int(round(color_float[2] * 255)),
    )
    return color_int


def _overlay_image_with_mask(
    image: PILImage,
    mask: np.ndarray,
    color_int: ColorInt,
    do_display_outlines_only: Optional[bool] = False,
    outline_width: Optional[int] = 3,
) -> PILImage:
    """Overlay an image with a mask.

    :param image: Image
    :param mask: Mask
    :param color_int: Color of the overlay in int format.
    :param do_display_outlines_only: If true, then only the outlines of the mask are used.
    :param outline_width: Width of the outlines.
    :return: PIL image with overlayed mask.
    """
    if do_display_outlines_only:
        mask = np.logical_xor(mask, binary_erosion(mask, iterations=outline_width))
    mask = PILImage.fromarray(mask)
    mask_colored = ImageOps.colorize(mask.convert("L"), black="black", white=color_int)
    image = PILImage.composite(mask_colored, image, mask)
    return image


def plot_particle_size_distributions(
    particle_size_lists: List[List[float]],
    score_lists: Optional[Iterable[List[float]]] = None,
    labels: Optional[Iterable[str]] = None,
    measurand_name: Optional[str] = "Diameter",
    unit: Optional[str] = "px",
) -> None:
    """Plot multiple particle size distributions.

    :param particle_size_lists: List of particle size lists.
    :param score_lists: List of score lists.
    :param labels: Labels for the different particle size distributions in the plot.
    :param measurand_name: Name of the measurand (used as x-label).
    :param unit: Unit of the measurement (used in x-label).
    """
    num_particle_size_distributions = len(particle_size_lists)
    colors = get_viridis_colors(num_particle_size_distributions)

    hist_kwargs = {"density": True, "histtype": "step"}

    if not labels:
        labels = [f"PSD {i}" for i in range(1, num_particle_size_distributions + 1)]

    if not score_lists:
        score_lists = [None] * num_particle_size_distributions

    bins = None

    for particle_sizes, scores, color, label in zip(
        particle_size_lists, score_lists, colors, labels
    ):
        geometric_mean = gmean(particle_sizes, weights=scores)
        geometric_standard_deviation = gstd(particle_sizes, weights=scores)

        label = (
            f"{label}\n  $d_g={geometric_mean:.0f}$ {unit}\n  $\sigma_g="
            f"{geometric_standard_deviation:.2f}$"
        )

        _, bins, _ = plt.hist(
            particle_sizes, bins=bins, color=color, weights=scores, label=label, **hist_kwargs
        )

    plt.ylabel("")
    plt.legend()

    plt.xlabel(f"{measurand_name}/{unit}")

    plt.ylabel("Probability Density")
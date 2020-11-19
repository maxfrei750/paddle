import random
import warnings
from statistics import gmean, gstd

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage.morphology import binary_erosion
from skimage import img_as_float, img_as_ubyte

import torch
from data import extract_bounding_box
from torchvision import transforms


def get_viridis_colors(num_colors):
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


def get_random_colors(num_colors):
    colors = list()

    for i_color in range(num_colors):
        # color = cm.hsv(random.uniform(0, 0.6))

        colormap = cm.get_cmap("viridis")
        color = colormap(random.uniform(0, 1))
        # Convert color to uint8.
        color = tuple([int(round(x * 255)) for x in color])
        colors.append(color)

    return colors


def display_detection(*args, **kwargs):
    result = visualize_detection(*args, **kwargs)
    result.show()


# TODO: Replace detection with individual arguments and adapt display_detection and save_visualization.
def visualize_detection(
    image,
    detection,
    do_display_box=True,
    do_display_outlines_only=True,
    do_display_label=True,
    do_display_score=True,
    do_display_mask=True,
    class_name_dict=None,
    score_threshold=0,
):
    font_size = 16

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.truetype("arial.ttf", font_size)

    if class_name_dict is None:
        class_name_dict = {1: "particle"}

    assert isinstance(image, torch.Tensor) or isinstance(
        image, np.ndarray
    ), "Expected image to be of class torch.Tensor or numpy.ndarray."

    image = image.squeeze()

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Convert from CWH to HWC if necessary.
    if not image.shape[2] == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = img_as_ubyte(image)
    image = transforms.ToPILImage()(image)

    for key in detection:
        value = detection[key]
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
            detection[key] = value

    for key in ["masks", "boxes"]:
        if key in detection.keys():
            num_instances = len(detection[key])
            break

        raise ValueError("Detection must have either masks or boxes.")

    if "masks" in detection:
        masks = detection["masks"]
    else:
        masks = [None] * num_instances

    if "boxes" in detection:
        boxes = detection["boxes"]
    else:
        boxes = [None] * num_instances

    if "scores" in detection:
        scores = detection["scores"]
    else:
        scores = [None] * num_instances

    if "labels" in detection:
        labels = detection["labels"]
    else:
        labels = [None] * num_instances

    result = image.convert("RGB")
    colors = get_random_colors(num_instances)

    for mask, box, color, score, label in zip(masks, boxes, colors, scores, labels):
        if score:
            if score <= score_threshold:
                continue

        if mask is not None and do_display_mask:
            mask = mask.squeeze()

            if mask.dtype == "uint8" and mask.max() <= 1:
                mask = img_as_float(mask * 255)
            mask = mask >= 0.5

            result = _overlay_image_with_mask(result, mask, color, do_display_outlines_only)

        if (do_display_label or do_display_score or do_display_box) and box is None:
            box = extract_bounding_box(mask)

        if box is not None and do_display_box:
            ImageDraw.Draw(result).rectangle(box, outline=color, width=2)

        caption = None

        if label is not None and do_display_label:
            caption = class_name_dict[label]

        if score is not None and do_display_score:
            if not do_display_label:
                caption = "Score"

            caption += ": {:.3f}".format(score)

        if label is not None and do_display_label or score is not None and do_display_score:
            x, y = box[:2]
            y -= font_size + 2
            ImageDraw.Draw(result).text((x, y), caption, font=font, fill=color)

    return result


def _overlay_image_with_mask(image, mask, color, do_display_outlines_only):
    if do_display_outlines_only:
        outline_width = 3
        mask = np.logical_xor(mask, binary_erosion(mask, iterations=outline_width))
    mask = Image.fromarray(mask)
    mask_colored = ImageOps.colorize(mask.convert("L"), black="black", white=color)
    image = Image.composite(mask_colored, image, mask)
    return image


def save_visualization(image, prediction, visualization_image_path, **kwargs):
    visualization_image = visualize_detection(image, prediction, **kwargs)
    visualization_image.save(visualization_image_path)


def plot_particle_size_distributions(
    particle_size_lists, score_lists=None, labels=None, measurand_name="Diameter", unit="px"
):
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

        label = f"{label}\n  $d_g={geometric_mean:.0f}$ {unit}\n  $\sigma_g={geometric_standard_deviation:.2f}$"

        _, bins, _ = plt.hist(
            particle_sizes, bins=bins, color=color, weights=scores, label=label, **hist_kwargs
        )

    plt.ylabel("")
    plt.legend()

    plt.xlabel(f"{measurand_name}/{unit}")

    plt.ylabel("Probability Density")

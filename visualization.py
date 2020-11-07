import random
import warnings

import numpy as np
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage.morphology import binary_erosion
from skimage import img_as_float, img_as_ubyte

import torch
from torchvision import transforms


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

    for key in ["masks", "boxes", "scores", "labels"]:
        if key in detection.keys():
            num_instances = len(detection[key])
            break

        raise ValueError("Detection does not have any visualizable information.")

    if "masks" in detection:
        masks = detection["masks"]
    else:
        masks = [None]

    if "boxes" in detection:
        boxes = detection["boxes"]
    else:
        boxes = [None]

    if "scores" in detection:
        scores = detection["scores"]
    else:
        scores = [None]

    if "labels" in detection:
        labels = detection["labels"]
    else:
        labels = [None]

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

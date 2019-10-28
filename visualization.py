from PIL import Image, ImageOps, ImageDraw
import pylab
import random
import numpy as np
from skimage.morphology import binary_erosion


def get_random_colors(n_colors, colormap_name="viridis", do_randomize=True):
    cm = pylab.get_cmap(colormap_name)

    colors = list()

    if do_randomize:
        n_colors_base = 255
        color_indices = random.sample(range(n_colors_base), n_colors)
    else:
        color_indices = range(n_colors)
        n_colors_base = n_colors

    for color_index in color_indices:
        color = cm(1. * color_index / n_colors_base)
        color = tuple([int(round(x * 255)) for x in color])
        colors.append(color)

    return colors


def display_detection(image, detection, do_display_box=True, do_display_outlines_only=True):
    masks = detection["masks"]

    n_instances = len(masks)

    boxes = detection["boxes"]

    if "scores" in detection:
        scores = detection["scores"]
    else:
        scores = [None] * n_instances

    labels = detection["labels"]

    result = image.convert("RGB")

    colors = get_random_colors(n_instances)

    for mask, box, color, score, label in zip(masks, boxes, colors, scores, labels):
        mask = mask.numpy()

        if do_display_outlines_only:
            mask = mask > 0
            mask = np.logical_xor(mask, binary_erosion(mask))

        mask = Image.fromarray(mask)
        mask = mask.convert("L")

        mask = ImageOps.colorize(mask, black="black", white=color)

        result = Image.blend(result, mask, 0.5)

        if do_display_box:
            box = box.numpy()
            ImageDraw.Draw(result).rectangle(box, outline=color)

    result.show()

        # TODO: Display scores and labels.

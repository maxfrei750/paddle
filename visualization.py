from PIL import Image, ImageOps, ImageDraw, ImageFont
import random
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from matplotlib import cm
from torchvision import transforms
import torch
from skimage import img_as_float, img_as_ubyte
import warnings
from spline import Spline


def get_random_colors(n_colors):
    colors = list()

    for i_color in range(n_colors):
        color = cm.hsv(random.uniform(0, 0.6))
        # Convert color to uint8.
        color = tuple([int(round(x * 255)) for x in color])
        colors.append(color)

    return colors


def display_detection(*args, **kwargs):
    result = visualize_detection(*args, **kwargs)
    result.show()


def visualize_detection(image,
                        detection,
                        do_display_box=True,
                        do_display_outlines_only=True,
                        do_display_label=True,
                        do_display_score=True,
                        do_display_mask=True,
                        do_display_keypoints=True,
                        do_display_keypoint_indices=True,
                        do_display_spline=True,
                        class_name_dict=None):
    font_size = 16

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.truetype("arial.ttf", font_size)

    if class_name_dict is None:
        class_name_dict = {
            1: "particle"
        }

    assert isinstance(image, torch.Tensor) or isinstance(image, np.ndarray), \
        "Expected image to be of class torch.Tensor or numpy.ndarray."

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

    if "masks" in detection:
        masks = detection["masks"]
        n_instances = len(masks)
    else:
        masks = [None]

    if "keypoints" in detection:
        key_point_sets = detection["keypoints"]
        n_instances = len(key_point_sets)
    else:
        key_point_sets = [None]

    if "spline_widths" in detection:
        spline_widths = detection["spline_widths"]
        n_instances = len(spline_widths)
    else:
        spline_widths = [None]

    if "boxes" in detection:
        boxes = detection["boxes"]
        n_instances = len(boxes)
    else:
        boxes = [None]

    if "scores" in detection:
        scores = detection["scores"]
        n_instances = len(scores)
    else:
        scores = [None]

    if "labels" in detection:
        labels = detection["labels"]
        n_instances = len(labels)
    else:
        labels = [None]

    result = image.convert("RGB")
    colors = get_random_colors(n_instances)

    for mask, box, color, score, label, key_points, spline_width in zip(masks,
                                                                        boxes,
                                                                        colors,
                                                                        scores,
                                                                        labels,
                                                                        key_point_sets,
                                                                        spline_widths):
        if mask is not None and do_display_mask:
            mask = mask.squeeze()

            if mask.dtype == "uint8" and mask.max() <= 1:
                mask = img_as_float(mask * 255)
            mask = mask >= 0.5

            result = _overlay_image_with_mask(result, mask, color, do_display_outlines_only)

        if key_points is not None and do_display_keypoints:
            x_list, y_list, is_visible_key_point_list = key_points.T
            point_size = 5
            r = point_size / 2

            for index, (x, y, is_visible_key_point) in enumerate(zip(x_list, y_list, is_visible_key_point_list)):
                if is_visible_key_point:
                    ImageDraw.Draw(result).ellipse([x - r, y - r, x + r, y + r], fill=color)

                    if do_display_keypoint_indices:
                        offset = 2
                        ImageDraw.Draw(result).text((x + offset, y + offset), str(index + 1), font=font, fill=color)

        if key_points is not None and spline_width is not None and do_display_spline:
            xy = key_points[:, :2]
            spline = Spline(xy, spline_width)
            spline_mask = spline.get_mask(result.size)

            result = _overlay_image_with_mask(result, spline_mask, color, do_display_outlines_only)

        if box is not None and do_display_box:
            ImageDraw.Draw(result).rectangle(box, outline=color, width=2)

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
        outline_width = 1
        mask = np.logical_xor(mask, binary_erosion(mask, iterations=outline_width))
    mask = Image.fromarray(mask)
    mask_colored = ImageOps.colorize(mask.convert("L"), black="black", white=color)
    image = Image.composite(mask_colored, image, mask)
    return image

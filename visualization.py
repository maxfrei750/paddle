from PIL import Image, ImageOps, ImageDraw, ImageFont
import random
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from matplotlib import cm
from torchvision import transforms
import torch
from skimage import img_as_float, img_as_ubyte
import warnings


def get_random_colors(n_colors):
    colors = list()

    for i_color in range(n_colors):
        color = cm.hsv(random.uniform(0, 0.6))
        # Convert color to uint8.
        color = tuple([int(round(x * 255)) for x in color])
        colors.append(color)

    return colors


def display_detection(image,
                      detection,
                      do_display_box=True,
                      do_display_outlines_only=True,
                      do_display_label=True,
                      do_display_score=True,
                      class_name_dict=None):

    result = visualize_detection(image,
                                 detection,
                                 do_display_box,
                                 do_display_outlines_only,
                                 do_display_label,
                                 do_display_score,
                                 class_name_dict)
    result.show()


def visualize_detection(image,
                        detection,
                        do_display_box=True,
                        do_display_outlines_only=True,
                        do_display_label=True,
                        do_display_score=True,
                        class_name_dict=None):
    if class_name_dict is None:
        class_name_dict = {
            1: "particle"
        }

    assert isinstance(image, torch.Tensor), "Expected image to be of class torch.Tensor."

    image = image.permute(1, 2, 0).cpu().numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = img_as_ubyte(image)
    image = transforms.ToPILImage()(image)

    for key in detection:
        value = detection[key]
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
            detection[key] = value

    masks = detection["masks"]
    boxes = detection["boxes"]
    scores = detection["scores"]
    labels = detection["labels"]
    n_instances = len(masks)
    result = image.convert("RGB")
    colors = get_random_colors(n_instances)

    for mask, box, color, score, label in zip(masks, boxes, colors, scores, labels):
        mask = mask.squeeze()

        if mask.dtype == "uint8" and mask.max() <= 1:
            mask = img_as_float(mask*255)
        mask = mask >= 0.5

        if do_display_outlines_only:
            outline_width = 1
            mask = np.logical_xor(mask, binary_erosion(mask, iterations=outline_width))

        mask = Image.fromarray(mask)

        mask_colored = ImageOps.colorize(mask.convert("L"), black="black", white=color)

        result = Image.composite(mask_colored, result, mask)

        if do_display_box:
            ImageDraw.Draw(result).rectangle(box, outline=color, width=2)

        if do_display_label:
            caption = class_name_dict[label]

        if do_display_score:
            if not do_display_label:
                caption = "Score"

            caption += ": {:.3f}".format(score)

        if do_display_label or do_display_score:
            font_size = 16
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
            x, y = box[:2]
            y -= font_size + 2
            ImageDraw.Draw(result).text((x, y), caption, font=font, fill=color)

    return result

import PIL
from PIL import Image, ImageOps, ImageDraw, ImageFont
import random
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from matplotlib import cm
from torchvision import  transforms


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
    if class_name_dict is None:
        class_name_dict = {
            1: "particle"
        }

    if not isinstance(image, PIL.Image.Image):
        image = transforms.ToPILImage()(image)

    masks = detection["masks"].numpy()

    n_instances = len(masks)

    boxes = detection["boxes"].numpy()

    scores = detection["scores"].numpy()

    labels = detection["labels"].numpy()

    result = image.convert("RGB")

    colors = get_random_colors(n_instances)

    for mask, box, color, score, label in zip(masks, boxes, colors, scores, labels):
        mask = mask > 0

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

    result.show()

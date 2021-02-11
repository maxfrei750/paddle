from pathlib import Path

import cv2
import fire
import numpy as np
import pandas as pd
from PIL import Image

from paddle.utilities import AnyPath


def convert_sopat_csv_to_masks(
    data_root: AnyPath, image_folder: AnyPath, csv_file: AnyPath, output_folder: AnyPath
) -> None:
    """Convert circular/elliptical ImageJ annotations to binary masks. Also converts and renames
        associated input images to png.

    :param data_root: Path, where ImageJ results csv-file and associated images are stored.
    :param image_folder: Path of the folder where the input images are stored, relative to data_root.
    :param csv_file: Path of the ImageJ results csv-file, relative to data_root.
    :param output_folder: Path of the folder, were resulting binary masks and images are stored,
        relative to data_root.
    """

    data_root = Path(data_root)
    output_path = data_root / output_folder
    output_path.mkdir(exist_ok=True, parents=True)

    csv_path = data_root / csv_file

    with open(csv_path, "r") as file:
        csv_data_raw = file.read()

    csv_blocks = csv_data_raw.split("FILE: ")[1:]

    for csv_block in csv_blocks:
        lines = csv_block.split("\n")
        lines = [line for line in lines if line != ""]

        image_file_name = lines[0].split("\\")[-1]

        # Extract relevant lines.
        annotations = lines[2:-2]
        annotations = [mask_parameters.split(";")[:-1] for mask_parameters in annotations]
        annotations = pd.DataFrame(annotations, columns=["r", "x", "y"])
        annotations = annotations.astype(float).astype(int)

        image_path = data_root / image_folder / image_file_name
        image_name = Path(image_file_name).stem
        image = Image.open(image_path)

        for annotation_index, annotation in annotations.iterrows():
            mask = np.zeros_like(image)

            center = (annotation["x"], annotation["y"])
            axes = (annotation["r"], annotation["r"])
            angle = 0

            mask = cv2.ellipse(mask, center, axes, angle, 0, 360, color=1, thickness=-1)
            mask = Image.fromarray(mask.astype(bool))
            mask.save(output_path / f"mask_{image_name}_{annotation_index}.png")


if __name__ == "__main__":
    fire.Fire(convert_sopat_csv_to_masks)

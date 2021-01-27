from pathlib import Path

import cv2
import fire
import numpy as np
import pandas as pd
from PIL import Image

from utilities import AnyPath


def convert_imagej_csv_to_masks(
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
    csv_data = pd.read_csv(csv_path)

    image_file_names = csv_data["Label"].unique()

    for image_file_name in image_file_names:
        annotation_data = csv_data[csv_data["Label"] == image_file_name]

        image_path = data_root / image_folder / image_file_name
        image_name = Path(image_file_name).stem
        image = Image.open(image_path)
        # image.save(output_path / f"image_{image_name}.png")

        for annotation_index, annotation in annotation_data.iterrows():
            mask = np.zeros_like(image)

            center = (int(annotation["X"]), int(annotation["Y"]))
            axes = (int(annotation["Major"] / 2), int(annotation["Minor"] / 2))
            angle = annotation["Angle"] + 90

            mask = cv2.ellipse(mask, center, axes, angle, 0, 360, color=1, thickness=-1)
            mask = Image.fromarray(mask.astype(bool))
            mask.save(output_path / f"mask_{image_name}_{annotation_index}.png")


if __name__ == "__main__":
    fire.Fire(convert_imagej_csv_to_masks)

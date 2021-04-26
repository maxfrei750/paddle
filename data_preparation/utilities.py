import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from data_preparation.sopat_catalyst import AnyPath


def split(data_root: AnyPath, validation_percentage: float, seed: int = 42):
    """Split data into a training and a validation set.

    :param data_root: Path, where images and mask folders are stored.
    :param validation_percentage: Percentage of images to put into the validation set.
    :param seed: random seed
    :return:
    """

    if not validation_percentage <= 1:
        raise ValueError("`validation_percentage` must be in the range [0, 1].")

    random.seed(seed)
    data_root = Path(data_root)

    validation_folder_path = data_root / "validation"
    validation_folder_path.mkdir(parents=True, exist_ok=True)

    training_folder_path = data_root / "training"
    training_folder_path.mkdir(parents=True, exist_ok=True)

    image_paths = list(data_root.glob("image_*.png"))
    num_images = len(image_paths)

    num_images_validation = round(validation_percentage * num_images)
    image_paths_validation = random.sample(image_paths, num_images_validation)

    _move_files(data_root, image_paths_validation, validation_folder_path)

    image_paths_training = data_root.glob("image_*.png")

    _move_files(data_root, image_paths_training, training_folder_path)


def _move_files(data_root, image_paths, subset_folder_path):
    for image_path in image_paths:
        image_id = image_path.stem[6:]
        mask_paths = data_root.glob(f"**/mask_{image_id}*.*")

        for mask_path in mask_paths:
            class_name = mask_path.parent.name
            new_mask_path = subset_folder_path / class_name / mask_path.name
            new_mask_path.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(mask_path, new_mask_path)

        shutil.move(image_path, subset_folder_path / image_path.name)


def convert_imagej_csv_to_masks_batch(
    data_root: AnyPath,
    image_folder: AnyPath,
    csv_file_glob: AnyPath,
    output_root: AnyPath,
    mask_folder_name: str,
) -> None:
    """Convert circular/elliptical ImageJ annotations to binary masks. Also converts and renames
        associated input images to png.

    :param data_root: Path, where ImageJ results csv-file and associated images are stored.
    :param image_folder: Path of the folder where the input images are stored, relative to data_root
    :param csv_file_glob: glob expression to identify ImageJ csv files, relative to the data_root.
    :param output_root: Path of the folder, were resulting binary masks and images are stored,
        relative to data_root.
    :param mask_folder_name: name of the folder, where masks are stored. Will be created in
        output_root.
    """

    data_root = Path(data_root)

    for csv_file_path in tqdm(data_root.glob(csv_file_glob)):
        convert_imagej_csv_to_masks(
            data_root, image_folder, csv_file_path, output_root, mask_folder_name
        )


def convert_imagej_csv_to_masks(
    data_root: AnyPath,
    image_folder: AnyPath,
    csv_path: AnyPath,
    output_root: AnyPath,
    mask_folder_name: str,
) -> None:
    """Convert circular/elliptical ImageJ annotations to binary masks. Also converts and renames
        associated input images to png.

    :param data_root: Path, where ImageJ results csv-file and associated images are stored.
    :param image_folder: Path of the folder where the input images are stored, relative to data_root
    :param csv_path: Path of the ImageJ results csv-file, relative to data_root.
    :param output_root: Path, where the output mask folder and images reside.
    :param mask_folder_name: name of the folder, where masks are stored. Will be created in
        output_root.
    """

    data_root = Path(data_root)
    output_root = Path(output_root)

    mask_folder_path = output_root / mask_folder_name
    mask_folder_path.mkdir(exist_ok=True, parents=True)

    csv_data = pd.read_csv(csv_path)

    image_file_names = csv_data["Label"].unique()

    for image_file_name in image_file_names:
        annotation_data = csv_data[csv_data["Label"] == image_file_name]

        image_path = data_root / image_folder / image_file_name
        image_name = Path(image_file_name).stem
        image = Image.open(image_path)
        image.save(output_root / f"image_{image_name}.png")

        for annotation_index, annotation in annotation_data.iterrows():
            mask = np.zeros_like(image)

            center = (int(annotation["X"]), int(annotation["Y"]))
            axes = (int(annotation["Major"] / 2), int(annotation["Minor"] / 2))
            angle = 360 - annotation["Angle"]

            mask = cv2.ellipse(mask, center, axes, angle, 0, 360, color=1, thickness=-1)
            mask = Image.fromarray(mask.astype(bool))
            mask.save(mask_folder_path / f"mask_{image_name}_{annotation_index}.png")

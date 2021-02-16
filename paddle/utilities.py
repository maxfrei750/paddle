import os
import random
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import requests
import torch
import yaml
from tqdm import tqdm

from .custom_types import AnyPath


def get_time_stamp() -> str:
    """Get current time formatted as string.

    :return: String representing the current time in the following format: %Y-%m-%d_%H-%M-%S
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_best_checkpoint_path(
    checkpoint_root: AnyPath,
    metric_key: Optional[str] = "val_mAP",
    mode: Literal["min", "max"] = "max",
) -> AnyPath:
    """Retrieve model checkpoint with the highest accuracy.


    :param checkpoint_root: Path where model checkpoints are saved.
    :param metric_key: Key to identify the relevant metric in the checkpoint filenames.
    :param mode: Criterion to select the best metric.
    :return: Path of the checkpoint with the highest accuracy.
    """

    expected_modes = ["min", "max"]
    assert mode in expected_modes, f"Expected parameter `mode` to be in {expected_modes}."

    checkpoint_root = Path(checkpoint_root)
    log_file_paths = list(checkpoint_root.glob("*.ckpt"))

    if not log_file_paths:
        raise FileNotFoundError(f"Could not find checkpoint file in: {checkpoint_root}")

    log_file_names = [Path(log_file_path.name).stem for log_file_path in log_file_paths]
    metric_values = [
        float(re.search(metric_key + r"=([+-]?((\d+\.?\d*)|(\.\d+)))", log_file_name)[1])
        for log_file_name in log_file_names
    ]

    if mode == "max":
        best_index = int(np.argmax(metric_values))
    else:
        best_index = int(np.argmin(metric_values))

    best_checkpoint_path = log_file_paths[best_index]
    return best_checkpoint_path


def get_latest_checkpoint_path(checkpoint_root: AnyPath) -> AnyPath:
    """Retrieve latest model checkpoint.

    :param checkpoint_root: Path where model checkpoints are saved.
    :return: Path of the last model checkpoint.
    """
    checkpoint_root = Path(checkpoint_root)
    log_file_paths = sorted(list(checkpoint_root.glob("*.ckpt")))

    if not log_file_paths:
        raise FileNotFoundError(f"Could not find checkpoint file in: {checkpoint_root}")

    return log_file_paths[-1]


def set_random_seed(seed: int):
    """Set the random seeds of the random, numpy and torch.

    :param seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_parameters_as_yaml(yaml_file_path: AnyPath, **kwargs: Any):
    """Store all optional parameters to a yaml file.

    :param yaml_file_path: Path of the output yaml file.
    :param kwargs: Dictionary of optional parameters
    """
    with open(yaml_file_path, "w") as file:
        yaml.dump(kwargs, file)


def get_latest_log_folder_path(log_root: AnyPath) -> AnyPath:
    """Get the id of the model that has been trained last.

    :param log_root: Path where log folders are located.
    :return: Id of the model that has been trained last.
    """
    log_root = Path(log_root)
    log_folders = [folder for folder in log_root.glob("*") if folder.is_dir()]
    last_log_folder = max(log_folders, key=os.path.getctime)
    last_model_id = last_log_folder.name
    return last_model_id


def download_file(url: str, output_file_path: AnyPath) -> None:
    """Download a file from the internet, if it does not exist yet.

    Based on:
    https://stackoverflow.com/a/37573701/11652760
    """
    output_file_path = Path(output_file_path)

    if output_file_path.is_file():
        warnings.warn(f"File {output_file_path} already exists. Skipping download.")
        return

    request_stream = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(request_stream.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    print(f"Downloading file from {url}...")

    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(output_file_path, "wb") as file:
        for data in request_stream.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Error while downloading checkpoint file.")

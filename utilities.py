import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml

import torch

AnyPath = Union[str, bytes, os.PathLike]


def get_time_stamp() -> str:
    """Get current time formatted as string.

    :return: String representing the current time in the following format: %Y-%m-%d_%H-%M-%S
    """
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_best_checkpoint_path(log_dir: AnyPath, filename_prefix: str) -> AnyPath:
    """Retrieve model checkpoint with the highest accuracy.

    :param log_dir: Path where model checkpoints are saved.
    :param filename_prefix: Checkpoint prefix.
    :return: Path of the checkpoint with the highest accuracy.
    """
    log_dir = Path(log_dir)
    log_file_paths = list(log_dir.glob(filename_prefix + "*.pt"))
    val_accuracies = [
        float(re.search(r"=([+-]?((\d+\.?\d*)|(\.\d+)))", str(log_file))[1])
        for log_file in log_file_paths
    ]
    best_index = int(np.argmax(val_accuracies))
    best_model_path = log_file_paths[best_index]
    return best_model_path


def get_last_checkpoint_path(log_dir: AnyPath) -> AnyPath:
    """Retrieve last model checkpoint.

    :param log_dir: Path where model checkpoints are saved.
    :return: Path of the last model checkpoint.
    """
    log_dir = Path(log_dir)
    log_file_paths = sorted(list(log_dir.glob("checkpoint_*_*.pt")))
    try:
        last_checkpoint_path = log_file_paths[-1]
    except IndexError:
        raise FileNotFoundError(f"Could not find checkpoint file in folder: {log_dir}")
    return last_checkpoint_path


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

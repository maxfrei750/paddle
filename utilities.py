import os
import random
import re
from datetime import datetime
from glob import glob
from os import path
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml


def get_time_stamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_best_model_path(log_dir, filename_prefix):
    log_files = glob(path.join(log_dir, filename_prefix + "*.pt"))
    val_accuracies = [
        float(re.search(r"=([+-]?((\d+\.?\d*)|(\.\d+)))", log_file)[1]) for log_file in log_files
    ]
    best_index = np.argmax(val_accuracies)
    best_model_path = log_files[best_index]
    return best_model_path


def get_last_checkpoint_path(log_dir: Union[str, bytes, os.PathLike]):
    log_dir = Path(log_dir)
    log_file_paths = sorted(list(log_dir.glob("checkpoint_*_*.pt")))
    last_checkpoint_path = log_file_paths[-1]
    return last_checkpoint_path


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_parameters_as_yaml(yaml_file_path, **kwargs):
    with open(yaml_file_path, "w") as file:
        yaml.dump(kwargs, file)

from os import PathLike
from typing import Dict, List, Tuple, TypedDict, Union

from numpy import ndarray
from torch import Tensor
from torch.optim import Optimizer

# General
AnyPath = Union[str, bytes, PathLike]

# Data
Mask = ndarray
Annotation = Dict[str, Union[Tensor, str]]  # TODO: Check if other types must be in the union.
Image = Tensor
Batch = Tuple[Tuple[Image, ...], Tuple[Annotation, ...]]

# Model
PartialLosses = Dict[str, Tensor]
Loss = Tensor


class ValidationOutput(TypedDict):
    predictions: List[Annotation]
    targets: Tuple[Annotation, ...]


class TestOutput(TypedDict):
    predictions: List[Annotation]
    targets: Tuple[Annotation, ...]
    images: Tuple[Image, ...]


class OptimizerConfiguration(TypedDict):
    optimizer: Optimizer
    lr_scheduler: object
    monitor: str


# Visualization
ColorFloat = Tuple[float, float, float]
ColorInt = Tuple[int, int, int]

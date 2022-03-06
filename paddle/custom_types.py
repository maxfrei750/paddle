from os import PathLike
from typing import Dict, List, Tuple, TypedDict, Union

from numpy import ndarray
from torch import Tensor
from torch.optim import Optimizer

# General
AnyPath = Union[str, PathLike]
ArrayLike = Union[Tensor, ndarray]

# Data
Mask = ArrayLike
Annotation = Dict[str, Union[ArrayLike, str]]
Image = ArrayLike
Batch = Tuple[Tuple[Image, ...], Tuple[Annotation, ...]]
CroppingRectangle = Tuple[int, int, int, int]

# Model
PartialLosses = Dict[str, Tensor]
Loss = Tensor


class ValidationOutput(TypedDict):
    predictions: List[Annotation]
    targets: Tuple[Annotation, ...]


class TestOutput(TypedDict):
    predictions: List[Annotation]


class OptimizerConfiguration(TypedDict):
    optimizer: Optimizer
    lr_scheduler: object
    monitor: str


# Visualization
ColorFloat = Tuple[float, float, float]
ColorInt = Tuple[int, int, int]

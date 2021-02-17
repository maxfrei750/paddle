from dataclasses import dataclass
from typing import Any


@dataclass
class TrainerConfigBase:
    fast_dev_run: bool = False
    gpus: Any = -1
    # TODO: Use Union[List[int], int, str] as soon as https://github.com/omry/omegaconf/issues/144
    #  is resolved.
    max_epochs: int = 300

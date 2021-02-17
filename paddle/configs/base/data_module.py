from dataclasses import dataclass
from typing import List, Optional

from omegaconf import MISSING


@dataclass
class DataModuleConfigBase:
    data_root: str = MISSING
    batch_size: int = 8
    train_subset: str = "training"
    val_subset: str = "validation"
    test_subset: Optional[str] = None
    cropping_rectangle: Optional[List[int]] = None

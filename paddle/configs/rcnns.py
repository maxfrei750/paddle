from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import BaseConfig


@dataclass
class RCNNConfig(BaseConfig):
    @dataclass
    class LightningModuleConfig(BaseConfig.LightningModuleConfig):
        num_classes: int = 2  # Background and Particle
        model_kwargs: Optional[Dict[str, Any]] = None
        # see torchvision.models.detection.mask_rcnn.MaskRCNN

    lightning_module: LightningModuleConfig = LightningModuleConfig()


@dataclass
class MaskRCNNConfig(RCNNConfig):
    pass


# TODO: Implement FibeRCNN
# @dataclass
# class FibeRCNNConfig(RCNNConfig):
#     pass

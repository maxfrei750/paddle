from dataclasses import dataclass
from typing import Any, Dict, Optional

from paddle.configs.base import ConfigBase, LightningModuleConfigBase


@dataclass
class LightningModuleConfigRCNN(LightningModuleConfigBase):
    num_classes: int = 2  # Background and Particle
    model_kwargs: Optional[Dict[str, Any]] = None
    # see torchvision.models.detection.mask_rcnn.MaskRCNN


@dataclass
class RCNNConfig(ConfigBase):
    lightning_module: LightningModuleConfigRCNN = LightningModuleConfigRCNN()


@dataclass
class MaskRCNNConfig(RCNNConfig):
    pass


# TODO: Implement FibeRCNN
# @dataclass
# class FibeRCNNConfig(RCNNConfig):
#     pass

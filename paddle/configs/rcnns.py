from dataclasses import dataclass
from typing import Any, Dict, Optional

from paddle.configs.base import ConfigBase, ModelConfigBase


@dataclass
class ModelConfigRCNN(ModelConfigBase):
    num_classes: int = 2  # Background and Particle
    model_kwargs: Optional[Dict[str, Any]] = None
    # see torchvision.models.detection.mask_rcnn.MaskRCNN


@dataclass
class RCNNConfig(ConfigBase):
    model: ModelConfigRCNN = ModelConfigRCNN()


@dataclass
class MaskRCNNConfig(RCNNConfig):
    pass


# TODO: Implement FibeRCNN
# @dataclass
# class FibeRCNNConfig(RCNNConfig):
#     pass

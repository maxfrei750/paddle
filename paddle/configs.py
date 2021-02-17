from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from hydra.conf import HydraConf, RunDir
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# TODO: Disable stdout log file.
# TODO: Refactor naming: remove all the redundant "Base" parts


@dataclass
class HydraRunConfigBase(RunDir):
    dir: str = "logs/${hydra:job.config_name}/${now:%Y-%m-%d_%H-%M-%S}"


@dataclass
class HydraConfigBase(HydraConf):
    run: HydraRunConfigBase = HydraRunConfigBase()


@dataclass
class ProgramConfigBase:
    random_seed: int = 42
    search_optimum_learning_rate: bool = False


@dataclass
class CallbackConfigBase:
    early_stopping_patience: int = 20
    # TODO: split into a section for each callback
    # TODO: Add all parameters of the callbacks


@dataclass
class TrainerConfigBase:
    fast_dev_run: bool = False
    gpus: Any = -1
    # TODO: Use Union[List[int], int, str] as soon as https://github.com/omry/omegaconf/issues/144
    #  is resolved.
    max_epochs: int = 300


@dataclass
class ModelConfigBase:
    num_classes: int = 2  # Background and Particle
    learning_rate: float = 0.005
    drop_lr_on_plateau_patience: int = 10
    model_kwargs: Optional[Dict[str, Any]] = None
    # see torchvision.models.detection.mask_rcnn.MaskRCNN


@dataclass
class DataModuleConfigBase:
    data_root: str = MISSING
    batch_size: int = 8
    train_subset: str = "training"
    val_subset: str = "validation"
    test_subset: Optional[str] = None
    cropping_rectangle: Optional[List[int]] = None


@dataclass
class RCNNConfigBase:
    hydra: HydraConfigBase = HydraConfigBase()
    program: ProgramConfigBase = ProgramConfigBase()
    callbacks: CallbackConfigBase = CallbackConfigBase()
    trainer: TrainerConfigBase = TrainerConfigBase()
    model: ModelConfigBase = ModelConfigBase()
    datamodule: DataModuleConfigBase = DataModuleConfigBase()


@dataclass
class MaskRCNNConfig(RCNNConfigBase):
    pass


def register() -> None:
    """Register configs."""
    cs = ConfigStore.instance()

    cs.store(name="MaskRCNN", node=MaskRCNNConfig, provider="paddle")

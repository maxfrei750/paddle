from dataclasses import dataclass

from .callbacks import CallbackConfigBase
from .datamodule import DataModuleConfigBase
from .hydra import HydraConfigBase
from .model import ModelConfigBase
from .program import ProgramConfigBase
from .trainer import TrainerConfigBase


@dataclass
class ConfigBase:
    hydra: HydraConfigBase = HydraConfigBase()
    program: ProgramConfigBase = ProgramConfigBase()
    callbacks: CallbackConfigBase = CallbackConfigBase()
    trainer: TrainerConfigBase = TrainerConfigBase()
    model: ModelConfigBase = ModelConfigBase()
    datamodule: DataModuleConfigBase = DataModuleConfigBase()

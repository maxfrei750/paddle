from dataclasses import dataclass

from .callbacks import CallbackConfigBase
from .data_module import DataModuleConfigBase
from .hydra import HydraConfigBase
from .lightning_module import LightningModuleConfigBase
from .program import ProgramConfigBase
from .trainer import TrainerConfigBase


@dataclass
class ConfigBase:
    hydra: HydraConfigBase = HydraConfigBase()
    program: ProgramConfigBase = ProgramConfigBase()
    callbacks: CallbackConfigBase = CallbackConfigBase()
    trainer: TrainerConfigBase = TrainerConfigBase()
    lightning_module: LightningModuleConfigBase = LightningModuleConfigBase()
    data_module: DataModuleConfigBase = DataModuleConfigBase()

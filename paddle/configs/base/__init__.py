from dataclasses import dataclass

from .callbacks import CallbackConfigBase
from .datamodule import DataModuleConfigBase
from .hydra import HydraConfigBase
from .lightning_module import LightningModuleConfigBase
from .program import ProgramConfigBase
from .trainer import TrainerConfigBase

# TODO: rename datamodule => data_module


@dataclass
class ConfigBase:
    hydra: HydraConfigBase = HydraConfigBase()
    program: ProgramConfigBase = ProgramConfigBase()
    callbacks: CallbackConfigBase = CallbackConfigBase()
    trainer: TrainerConfigBase = TrainerConfigBase()
    lightning_module: LightningModuleConfigBase = LightningModuleConfigBase()
    datamodule: DataModuleConfigBase = DataModuleConfigBase()

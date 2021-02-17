from dataclasses import dataclass


@dataclass
class LightningModuleConfigBase:
    learning_rate: float = 0.005
    drop_lr_on_plateau_patience: int = 10

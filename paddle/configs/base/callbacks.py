from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStoppingConfigBase:
    monitor: str = "val/mAP"
    patience: int = 20
    mode: str = "max"


@dataclass
class LearningRateMonitorConfigBase:
    logging_interval: Optional[str] = None
    log_momentum: bool = False


@dataclass
class ModelCheckpointConfigBase:
    monitor: Optional[str] = "val/mAP"
    mode: str = "max"
    filename: Optional[str] = "{epoch}-{step}-{val/mAP:.4f}"


@dataclass
class CallbackConfigBase:
    early_stopping: EarlyStoppingConfigBase = EarlyStoppingConfigBase()
    learning_rate_monitor: LearningRateMonitorConfigBase = LearningRateMonitorConfigBase()
    model_checkpoint: ModelCheckpointConfigBase = ModelCheckpointConfigBase()

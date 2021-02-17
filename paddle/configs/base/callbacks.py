from dataclasses import dataclass
from typing import Optional


@dataclass
class EarlyStoppingConfigBase:
    monitor: str = "val/mAP"
    patience: int = 20
    mode: str = "max"


@dataclass
class ModelCheckpointConfigBase:
    monitor: Optional[str] = "val/mAP"
    mode: str = "max"
    filename: Optional[str] = "{epoch}-{step}-{val/mAP:.4f}"


@dataclass
class ExampleDetectionMonitorBase:
    score_threshold: float = 0
    do_display_box: Optional[bool] = True
    do_display_label: Optional[bool] = True
    do_display_score: Optional[bool] = True
    do_display_mask: Optional[bool] = True
    do_display_outlines_only: Optional[bool] = True
    line_width: Optional[int] = 1
    font_size: Optional[int] = 16


@dataclass
class CallbackConfigBase:
    early_stopping: EarlyStoppingConfigBase = EarlyStoppingConfigBase()
    model_checkpoint: ModelCheckpointConfigBase = ModelCheckpointConfigBase()
    example_detection_monitor: ExampleDetectionMonitorBase = ExampleDetectionMonitorBase()

from dataclasses import dataclass
from typing import Any, List, Optional

from hydra.conf import HydraConf, RunDir
from omegaconf import MISSING

# TODO: Add docstrings.


@dataclass
class BaseConfig:
    @dataclass
    class HydraConfig(HydraConf):
        @dataclass
        class HydraRunConfigBase(RunDir):
            dir: str = "logs/${hydra:job.config_name}/${now:%Y-%m-%d_%H-%M-%S}"

        run: HydraRunConfigBase = HydraRunConfigBase()

    hydra: HydraConfig = HydraConfig()

    @dataclass
    class ProgramConfig:
        random_seed: int = 42
        search_optimum_learning_rate: bool = False

    program: ProgramConfig = ProgramConfig()

    @dataclass
    class CallbackConfig:
        @dataclass
        class EarlyStoppingConfig:
            monitor: str = "val/mAP"
            patience: int = 20
            mode: str = "max"

        early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()

        @dataclass
        class ModelCheckpointConfig:
            monitor: Optional[str] = "val/mAP"
            mode: str = "max"
            filename: Optional[str] = "{epoch}-{step}-{val/mAP:.4f}"

        model_checkpoint: ModelCheckpointConfig = ModelCheckpointConfig()

        @dataclass
        class ExampleDetectionMonitor:
            score_threshold: float = 0
            do_display_box: Optional[bool] = True
            do_display_label: Optional[bool] = True
            do_display_score: Optional[bool] = True
            do_display_mask: Optional[bool] = True
            do_display_outlines_only: Optional[bool] = True
            line_width: Optional[int] = 1
            font_size: Optional[int] = 16

        example_detection_monitor: ExampleDetectionMonitor = ExampleDetectionMonitor()

    callbacks: CallbackConfig = CallbackConfig()

    @dataclass
    class TrainerConfig:
        fast_dev_run: bool = False
        max_epochs: int = 300
        gpus: Any = -1
        # TODO: Use Union[List[int], int, str] as soon as
        #  https://github.com/omry/omegaconf/issues/144 is resolved.

    trainer: TrainerConfig = TrainerConfig()

    @dataclass
    class LightningModuleConfig:
        learning_rate: float = 0.005
        drop_lr_on_plateau_patience: int = 10

    lightning_module: LightningModuleConfig = LightningModuleConfig()

    @dataclass
    class DataModuleConfig:
        data_root: str = MISSING
        batch_size: int = 8
        train_subset: str = "training"
        val_subset: str = "validation"
        test_subset: Optional[str] = None
        initial_cropping_rectangle: Optional[List[int]] = None

    data_module: DataModuleConfig = DataModuleConfig()

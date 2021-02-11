import logging
import os
import warnings
from pathlib import Path
from typing import Tuple

import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from paddle.callbacks import ExampleDetectionMonitor, ModelCheckpoint
from paddle.data import MaskRCNNDataModule
from paddle.models import LightningMaskRCNN
from paddle.utilities import AnyPath


@hydra.main(config_path="configs", config_name="maskrcnn")
def train_model(config: DictConfig) -> None:
    """Trains a Mask R-CNN based on a given config.

    :param config: OmegaConf dictionary.
    """

    # TODO: Remove this filter, as soon as torchvision>0.8.2 is released.
    warnings.filterwarnings(
        "ignore",
        message="The default behavior for interpolate/upsample with float scale_factor changed",
    )

    log_root, version = setup_hydra()

    logging.getLogger(__name__).info(
        f"Training with the following config:\n{OmegaConf.to_yaml(config)}"
    )

    seed_everything(config.program.random_seed)

    data_module = MaskRCNNDataModule(**config.datamodule)

    model = LightningMaskRCNN(**config.model)

    if config.program.search_optimum_learning_rate:
        lr_tuner = Trainer(auto_lr_find=True, **config.trainer)
        lr_tuner.tune(model, datamodule=data_module)
        exit()

    callbacks = [
        ModelCheckpoint(monitor="val/mAP", mode="max", filename="{epoch}-{step}-{val/mAP:.4f}"),
        EarlyStopping(
            monitor="val/mAP", mode="max", patience=config.callbacks.early_stopping_patience
        ),
        LearningRateMonitor(),
        ExampleDetectionMonitor(),
    ]

    trainer = Trainer(
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=str(log_root), name="", version=version),
        **config.trainer,
    )

    trainer.fit(model, datamodule=data_module)


def setup_hydra() -> Tuple[AnyPath, str]:
    """Hydra automatically changes the working directory to the logging directory. This function
        extracts the log_root and version string needed by lightning and restores the original
        working directory.

    :return: Path where logs are stored and version string.
    """
    log_dir = Path.cwd()
    log_root = log_dir.parent
    version = log_dir.name
    os.chdir(hydra.utils.get_original_cwd())
    return log_root, version


if __name__ == "__main__":
    train_model()

import logging
import os
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from callbacks import ExampleDetectionMonitor
from data import MaskRCNNDataModule
from models import LightningMaskRCNN

# TODO: Docstrings
# TODO: Use typing.
# TODO: Remove obsolete functions remaining from ignite-lightning transition.


@hydra.main(config_path="configs", config_name="maskrcnn")
def train(config: DictConfig) -> None:
    log_dir = Path.cwd()
    log_root = log_dir.parent
    version = log_dir.name

    os.chdir(hydra.utils.get_original_cwd())

    logger = logging.getLogger(__name__)
    logger.info(f"Training with the following config:\n{OmegaConf.to_yaml(config)}")

    pl.seed_everything(config.program.random_seed)

    data_module = MaskRCNNDataModule(**config.datamodule)

    model = LightningMaskRCNN(**config.model)

    if config.program.search_optimum_learning_rate:
        lr_tuner = pl.Trainer(auto_lr_find=True)
        lr_tuner.tune(model, datamodule=data_module)
        exit()

    callbacks = [
        ModelCheckpoint(monitor="val/mAP", mode="max"),
        EarlyStopping(
            monitor="val/mAP", mode="max", patience=config.callbacks.early_stopping_patience
        ),
        LearningRateMonitor(),
        ExampleDetectionMonitor(),
    ]

    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=str(log_root), name="", version=version),
        **config.trainer,
    )

    # TODO: Remove this filter, as soon as torchvision>0.8.2 is released.
    warnings.filterwarnings(
        "ignore",
        message="The default behavior for interpolate/upsample with float scale_factor changed",
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    train()

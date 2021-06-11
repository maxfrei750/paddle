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

try:
    import wandb
except ImportError:
    wandb = None


from .callbacks.training import ExampleDetectionMonitor, ModelCheckpoint
from .custom_types import AnyPath
from .data import MaskRCNNDataModule
from .lightning_modules import LightningMaskRCNN


@hydra.main(config_name="MaskRCNN")
def train_mask_rcnn(config: DictConfig) -> None:
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

    setup_wandb(config, version)

    seed_everything(config.program.random_seed)

    data_module = MaskRCNNDataModule(**config.data_module)

    model = LightningMaskRCNN(**config.lightning_module)

    if config.program.search_optimum_learning_rate:
        lr_tuner = Trainer(auto_lr_find=True, **config.trainer)
        lr_tuner.tune(model, datamodule=data_module)
        exit()

    callbacks = [
        ModelCheckpoint(**config.callbacks.model_checkpoint),
        EarlyStopping(**config.callbacks.early_stopping),
        LearningRateMonitor(),
        ExampleDetectionMonitor(**config.callbacks.example_detection_monitor),
    ]

    trainer = Trainer(
        callbacks=callbacks,
        logger=TensorBoardLogger(save_dir=str(log_root), name="", version=version),
        **config.trainer,
    )

    trainer.fit(model, datamodule=data_module)


def setup_wandb(config, version):
    if config.logging.wandb.use:
        if config.trainer.fast_dev_run:
            warnings.warn("Not using wandb, since config.trainer.fast_dev_run=True.")
            return

        if wandb is None:
            raise ImportError(
                "There was an error during the import of wandb. Please check if it is installed."
            )

        wandb.init(
            sync_tensorboard=True,
            config=OmegaConf.to_container(config, resolve=True),
            name=version,
            project=config.logging.wandb.project_name,
        )


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

from os import path

from tensorboardX import SummaryWriter

import albumentations
import torch
from config import Config
from data import get_data_loader
from ignite.engine import Events, create_supervised_evaluator
from metrics import AveragePrecision
from models import get_model
from torchvision_detection_references.utils import collate_fn
from training import (
    create_trainer,
    get_lr_scheduler,
    get_optimizer,
    setup_checkpointers,
    setup_logging_callbacks,
)
from utilities import get_time_stamp, set_random_seed


def main():
    # Parameters -------------------------------------------------------------------------------------------------------
    test_mode = False
    config_name = "maskrcnn"
    log_dir_base = "logs"
    data_root = path.join("data")

    # Config -----------------------------------------------------------------------------------------------------------
    config = Config.load(path.join("configs", config_name + ".yml"))

    # Testmode ---------------------------------------------------------------------------------------------------------
    if test_mode:
        config["data"]["subset_training"] += "_mini"
        config["data"]["subset_validation"] += "_mini"

    # Reproducibility --------------------------------------------------------------------------------------------------
    set_random_seed(config["general"]["random_seed"])

    # Device -----------------------------------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model ------------------------------------------------------------------------------------------------------------
    model = get_model(
        num_classes=config["model"]["n_classes"],
        pretrained_backbone=config["model"]["pretrained_backbone"],
        trainable_backbone_layers=config["model"]["trainable_backbone_layers"],
    )

    # Data -------------------------------------------------------------------------------------------------------------
    # TODO: Test pillow-SIMD

    data_loader_training = get_data_loader(
        data_root,
        subset=config["data"]["subset_training"],
        batch_size=config["data"]["batch_size_training"],
        class_names=config["data"]["class_names"],
        num_workers=config["data"]["n_data_loader_workers"],
        transforms=get_transform(),
        collate_fn=collate_fn,
    )

    data_loader_validation = get_data_loader(
        data_root,
        subset=config["data"]["subset_validation"],
        batch_size=config["data"]["batch_size_validation"],
        class_names=config["data"]["class_names"],
        num_workers=config["data"]["n_data_loader_workers"],
        transforms=get_transform(),
        collate_fn=collate_fn,
    )

    # Optimizer --------------------------------------------------------------------------------------------------------
    optimizer = get_optimizer(
        model,
        optimizer_name=config["optimizer"]["name"],
        optimizer_parameters=config["optimizer"]["parameters"],
    )

    # Trainer ----------------------------------------------------------------------------------------------------------
    trainer = create_trainer(model, optimizer, data_loader_training, device)

    # Learning rate scheduler ------------------------------------------------------------------------------------------
    lr_scheduler = get_lr_scheduler(optimizer, config["lr_scheduler"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def step_lr_scheduler(engine):
        lr_scheduler.step()

    # Evaluation -------------------------------------------------------------------------------------------------------
    metrics = {"AP": AveragePrecision(data_loader_validation, device)}

    evaluator_validation = create_supervised_evaluator(model, metrics=metrics, device=device)

    # Logging ----------------------------------------------------------------------------------------------------------
    time_stamp = get_time_stamp()
    log_dir = path.join(log_dir_base, config_name + "_" + time_stamp)

    config.save(path.join(log_dir, config_name + ".yml"))

    tensorboard_writer = SummaryWriter(log_dir=log_dir, max_queue=0, flush_secs=20)
    setup_logging_callbacks(
        model,
        device,
        data_loader_validation,
        tensorboard_writer,
        evaluator_validation,
        metrics,
        trainer,
        config["logging"]["print_frequency"],
    )

    # Checkpointers ----------------------------------------------------------------------------------------------------
    setup_checkpointers(model, log_dir, trainer, evaluator_validation)

    # Training ---------------------------------------------------------------------------------------------------------
    try:
        trainer.run(data_loader_training, max_epochs=config["training"]["max_epochs"])
    finally:
        pass
        tensorboard_writer.close()

    # TODO: Early stopping


def get_transform():
    transform = albumentations.Compose(
        [
            albumentations.RandomCrop(width=1024, height=1024),
            albumentations.HorizontalFlip(0.5),
            albumentations.VerticalFlip(0.5),
            albumentations.RandomRotate90(0.75),
        ]
    )

    return transform


if __name__ == "__main__":
    main()

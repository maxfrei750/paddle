import torch
from torchvision_detection_references.utils import collate_fn
import torchvision_detection_references.transforms as T
from data import get_data_loaders
from os import path
from utilities import get_time_stamp, set_random_seed
from models import get_model
from training import create_trainer, get_optimizer, get_lr_scheduler, setup_logging_callbacks
from ignite.engine import create_supervised_evaluator, Events
from metrics import AveragePrecision
from tensorboardX import SummaryWriter
from ignite.handlers import ModelCheckpoint
from config import Config


def main():
    # Parameters -------------------------------------------------------------------------------------------------------
    test_mode = True
    config_file_name = "mrcnn.yml"

    # Config -----------------------------------------------------------------------------------------------------------
    config = Config.load(path.join("configs", config_file_name))

    # Testmode ---------------------------------------------------------------------------------------------------------
    if test_mode:
        config["data"]["subset_train"] += "_mini"
        config["data"]["subset_val"] += "_mini"

    # Reproducibility --------------------------------------------------------------------------------------------------
    set_random_seed(config["general"]["random_seed"])

    # Device -----------------------------------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model ------------------------------------------------------------------------------------------------------------
    model = get_model(config)

    # Paths ------------------------------------------------------------------------------------------------------------
    time_stamp = get_time_stamp()
    log_dir = path.join("logs", model.name + "_" + time_stamp)
    data_root = path.join("datasets", "IUTA", "easy_images", "individual_fibers_no_clutter_no_loops")

    # Data -------------------------------------------------------------------------------------------------------------
    # TODO: Test pillow-SIMD
    data_loader_train, data_loader_val = \
        get_data_loaders(data_root, config, collate_fn=collate_fn, transforms=get_transform())

    # Tensorboard ------------------------------------------------------------------------------------------------------
    tensorboard_writer = SummaryWriter(log_dir=log_dir, max_queue=0, flush_secs=20)

    # Optimizer --------------------------------------------------------------------------------------------------------
    optimizer = get_optimizer(model, config)

    # Trainer ----------------------------------------------------------------------------------------------------------
    trainer = create_trainer(model, optimizer, data_loader_train, device)

    # Learning rate scheduler ------------------------------------------------------------------------------------------
    lr_scheduler = get_lr_scheduler(optimizer, config)

    @trainer.on(Events.EPOCH_COMPLETED)
    def step_lr_scheduler(engine):
        lr_scheduler.step()

    # Evaluation -------------------------------------------------------------------------------------------------------
    metrics = {
        "AP": AveragePrecision(data_loader_val, device)
    }

    evaluator_val = create_supervised_evaluator(model, metrics=metrics, device=device)

    # Logging ----------------------------------------------------------------------------------------------------------
    config.save(path.join(log_dir, "config_" + model.name + ".yml"))
    setup_logging_callbacks(model, config, device, data_loader_val, tensorboard_writer, evaluator_val, metrics, trainer)

    # Checkpointers ----------------------------------------------------------------------------------------------------
    best_model_saver = ModelCheckpoint(log_dir,
                                       filename_prefix="model",
                                       score_name="AP",
                                       score_function=score_function,
                                       n_saved=1,
                                       atomic=True,
                                       create_dir=True)
    evaluator_val.add_event_handler(Events.COMPLETED, best_model_saver, {model.name: model})

    last_model_saver = ModelCheckpoint(log_dir,
                                       filename_prefix="checkpoint",
                                       save_interval=1,
                                       n_saved=1,
                                       atomic=True,
                                       create_dir=True)
    trainer.add_event_handler(Events.COMPLETED, last_model_saver, {model.name: model})

    # Training ---------------------------------------------------------------------------------------------------------
    try:
        trainer.run(data_loader_train, max_epochs=config["training"]["max_epochs"])
    finally:
        pass
        tensorboard_writer.close()

    # TODO: Early stopping


def score_function(engine):
    return engine.state.metrics["AP"]


def get_transform():
    transforms = list()
    transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)


if __name__ == "__main__":
    main()

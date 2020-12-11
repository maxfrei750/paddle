import math
import sys
from os import path

import numpy as np
import torch
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint
from tensorboardX import SummaryWriter

import torchvision_detection_references.utils as utils
from data import get_data_loader
from metrics import AveragePrecision
from models import get_model
from torchvision_detection_references.utils import collate_fn
from transforms import get_transform
from utilities import get_time_stamp, set_random_seed
from visualization import visualize_detection


def create_trainer(model, optimizer, data_loader, device=None):
    if device:
        model.to(device)

    def _update(engine, batch):
        epoch = engine.state.epoch

        model.train()

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1.0 / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        images, targets = batch

        images = list(image.to(device) for image in images)
        targets = [
            {k: v.to(device) for k, v in t.items() if isinstance(v, torch.Tensor)} for t in targets
        ]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        # TODO (optional): implement gradient norm
        #  torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        output_dict = dict()

        for key in loss_dict_reduced:
            output_dict[key] = loss_dict_reduced[key].cpu().detach().numpy().item()

        output_dict["loss"] = loss_value

        output_dict["lr"] = optimizer.param_groups[0]["lr"]

        return output_dict

    return Engine(_update)


def get_optimizer(model, optimizer_name, optimizer_parameters):
    expected_optimizer_names = ["sgd", "adam"]
    assert (
        optimizer_name in expected_optimizer_names
    ), f"Unknown optimizer name '{optimizer_name}'. Expected optimizer name to be in {expected_optimizer_names}."

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "sgd":
        return torch.optim.SGD(trainable_parameters, **optimizer_parameters)
    elif optimizer_name == "adam":
        return torch.optim.Adam(trainable_parameters, **optimizer_parameters)


def get_lr_scheduler(optimizer, lr_scheduler_parameters):
    return torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_parameters)


def setup_logging_callbacks(
    model,
    device,
    data_loader_val,
    tensorboard_writer,
    evaluator_val,
    metrics,
    trainer,
    print_frequency,
):
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_summary(engine):
        epoch = engine.state.epoch

        # Training
        for key in engine.state.output:
            tag = "training/" + key
            tensorboard_writer.add_scalar(tag, engine.state.output[key], epoch)

        # Validation
        print(" Validation:")

        evaluator_val.run(data_loader_val)
        metrics["AP"].print()
        tensorboard_writer.add_scalar("validation/AP", metrics["AP"].value, epoch)

        example_image = next(iter(data_loader_val))[0][0]
        example_image = example_image.to(device)

        model.eval()
        with torch.no_grad():
            prediction = model([example_image])[0]

        detection_image = np.array(
            visualize_detection(example_image, prediction, score_threshold=0.5)
        )
        tensorboard_writer.add_image(
            "validation/example_detection", detection_image, epoch, dataformats="HWC"
        )

    @trainer.on(Events.EPOCH_STARTED)
    def log_epoch(engine):
        epoch = engine.state.epoch

        print("\nEpoch: {}".format(epoch))
        print(" Training:")

        if epoch == 1:
            engine.state.previous_epoch = 0

        if engine.state.previous_epoch != epoch:
            engine.state.epoch_iteration = 0
            engine.state.previous_epoch = epoch

    @trainer.on(Events.ITERATION_STARTED)
    def increment_epoch_iteration(engine):
        engine.state.epoch_iteration += 1

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_losses(engine):
        epoch_iteration = engine.state.epoch_iteration

        if (epoch_iteration - 1) % print_frequency == 0:
            output = engine.state.output
            delimiter = "    "

            log_items = list()
            log_items.append(" [{:4d}]".format(epoch_iteration))
            log_items += ["{}: {:.4f}".format(key, output[key]) for key in output]

            log_msg = delimiter.join(log_items)
            print(log_msg)


def setup_checkpointers(model, log_dir, trainer, evaluator_val):
    def score_function(engine):
        return engine.state.metrics["AP"]

    best_model_saver = ModelCheckpoint(
        log_dir,
        filename_prefix="model",
        score_name="AP",
        score_function=score_function,
        n_saved=1,
        atomic=True,
        create_dir=True,
    )
    evaluator_val.add_event_handler(Events.COMPLETED, best_model_saver, {"MaskRCNN": model})

    last_model_saver = ModelCheckpoint(
        log_dir, filename_prefix="checkpoint", n_saved=1, atomic=True, create_dir=True
    )
    trainer.add_event_handler(Events.COMPLETED, last_model_saver, {"MaskRCNN": model})


def training(config):
    set_random_seed(config["general"]["random_seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(n_classes=config["model"]["n_classes"])
    data_loader_training, data_loader_validation = get_data_loaders(config)
    optimizer = get_optimizer(
        model,
        optimizer_name=config["optimizer"]["name"],
        optimizer_parameters=config["optimizer"]["parameters"],
    )
    trainer = get_trainer(config, data_loader_training, device, model, optimizer)
    tensorboard_writer = setup_logging(config, data_loader_validation, device, model, trainer)

    try:
        trainer.run(data_loader_training, max_epochs=config["training"]["max_epochs"])
    finally:
        tensorboard_writer.close()
    # TODO: Early stopping


def get_data_loaders(config):
    data_loader_training = get_data_loader(
        config["data"]["root_folder"],
        subset=config["data"]["subset_training"],
        batch_size=config["data"]["batch_size_training"],
        class_names=config["data"]["class_names"],
        num_workers=config["data"]["n_data_loader_workers"],
        transforms=get_transform(
            training=True, cropping_rectangle=config["data"]["cropping_rectangle"]
        ),
        collate_fn=collate_fn,
    )
    data_loader_validation = get_data_loader(
        config["data"]["root_folder"],
        subset=config["data"]["subset_validation"],
        batch_size=config["data"]["batch_size_validation"],
        class_names=config["data"]["class_names"],
        num_workers=config["data"]["n_data_loader_workers"],
        transforms=get_transform(
            training=False, cropping_rectangle=config["data"]["cropping_rectangle"]
        ),
        collate_fn=collate_fn,
    )
    return data_loader_training, data_loader_validation


def get_trainer(config, data_loader_training, device, model, optimizer):
    trainer = create_trainer(model, optimizer, data_loader_training, device)
    # Learning rate scheduler ------------------------------------------------------------------------------------------
    lr_scheduler = get_lr_scheduler(optimizer, config["lr_scheduler"])

    @trainer.on(Events.EPOCH_COMPLETED)
    def step_lr_scheduler(engine):
        lr_scheduler.step()

    return trainer


def setup_logging(config, data_loader_validation, device, model, trainer):
    metrics = {"AP": AveragePrecision(data_loader_validation, device)}
    evaluator_validation = create_supervised_evaluator(model, metrics=metrics, device=device)
    # Logging ----------------------------------------------------------------------------------------------------------
    time_stamp = get_time_stamp()
    log_dir_base = config["logging"]["root_folder"]
    log_dir = path.join(log_dir_base, config["name"] + "_" + time_stamp)
    config.save(path.join(log_dir, config["name"] + ".yml"))
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
    return tensorboard_writer

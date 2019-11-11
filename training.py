import math
import sys
from ignite.engine import Engine
import torchvision_detection_references.utils as utils
import torch
from visualization import visualize_detection
import numpy as np
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint


def create_trainer(model, optimizer, data_loader, device=None):
    if device:
        model.to(device)

    def _update(engine, batch):
        epoch = engine.state.epoch

        model.train()

        lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(data_loader) - 1)

            lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

        images, targets = batch

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

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


def get_optimizer(model, config):
    optimizer_name = config["optimizer"]["name"].lower()

    expected_optimizer_names = ["sgd", "adam"]
    assert optimizer_name in expected_optimizer_names, \
        f"Unknown optimizer name '{optimizer_name}'. Expected optimizer name to be in {expected_optimizer_names}."

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]

    if optimizer_name == "sgd":
        return torch.optim.SGD(trainable_parameters, **config["optimizer"]["parameters"])
    elif optimizer_name == "adam":
        return torch.optim.Adam(trainable_parameters, **config["optimizer"]["parameters"])


def get_lr_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.StepLR(optimizer, **config["lr_scheduler"])


def setup_logging_callbacks(model, config, device, data_loader_val, tensorboard_writer, evaluator_val, metrics, trainer):
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_epoch_summary(engine):
        epoch = engine.state.epoch

        # Training
        for key in engine.state.output:
            tag = "training/" + key
            tensorboard_writer.add_scalar(tag, engine.state.output[key], epoch)

        # Validation
        print(" Validation:")
        if model.name == "mrcnn":
            evaluator_val.run(data_loader_val)
            metrics["AP"].print()
            tensorboard_writer.add_scalar("validation/AP", metrics["AP"].value, epoch)
        elif model.name == "krcnn":
            # TODO: Write evaluator.
            pass

        example_image = next(iter(data_loader_val))[0][0]
        example_image = example_image.to(device)

        model.eval()
        with torch.no_grad():
            prediction = model([example_image])[0]

        detection_image = np.array(visualize_detection(example_image, prediction))
        tensorboard_writer.add_image("validation/example_detection", detection_image, epoch, dataformats="HWC")

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

        if (epoch_iteration - 1) % config["logging"]["print_frequency"] == 0:
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

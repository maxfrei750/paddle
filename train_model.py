import torch
from torch.utils.data import DataLoader
from torchvision_detection_references.utils import collate_fn
import torchvision_detection_references.transforms as T
from dataset import Dataset
from os import path
from utilities import get_time_stamp, set_random_seed
from models import get_model
from training import create_trainer
from ignite.engine import create_supervised_evaluator, Events
from metrics import AveragePrecision
from tensorboardX import SummaryWriter
from ignite.handlers import ModelCheckpoint
from visualization import visualize_detection
import numpy as np
from config import Config


def main():
    # Config -----------------------------------------------------------------------------------------------------------
    config_file_name = "mrcnn"
    config = Config.load(path.join("configs", config_file_name + ".yml"))

    # Testmode ---------------------------------------------------------------------------------------------------------
    test_mode = True

    if test_mode:
        config["subset_train"] += "_mini"
        config["subset_val"] += "_mini"

    # Reproducibility --------------------------------------------------------------------------------------------------
    set_random_seed(config["random_seed"])

    # Model ------------------------------------------------------------------------------------------------------------
    model = get_model(config)

    # Paths ------------------------------------------------------------------------------------------------------------
    time_stamp = get_time_stamp()
    log_dir = path.join("logs", model.name + "_" + time_stamp)
    data_root = path.join("datasets", "IUTA", "easy_images", "individual_fibers_no_clutter_no_loops")

    # Data -------------------------------------------------------------------------------------------------------------
    # TODO: Test pillow-SIMD
    data_loader_train, data_loader_val = \
        get_data_loaders(data_root,
                         subset_train=config["subset_train"], subset_val=config["subset_val"],
                         batch_size_train=config["batch_size_train"], batch_size_val=config["batch_size_val"])

    # Tensorboard ------------------------------------------------------------------------------------------------------
    tensorboard_writer = SummaryWriter(log_dir=log_dir, max_queue=0, flush_secs=20)

    # Device -----------------------------------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Optimizer --------------------------------------------------------------------------------------------------------
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(parameters, lr=1e-3)

    # Trainer ----------------------------------------------------------------------------------------------------------
    trainer = create_trainer(model, optimizer, data_loader_train, device)

    # Learning rate scheduler ------------------------------------------------------------------------------------------
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    @trainer.on(Events.EPOCH_COMPLETED)
    def step_lr_scheduler(engine):
        lr_scheduler.step()

    # Evaluation -------------------------------------------------------------------------------------------------------
    metrics = {
        "AP": AveragePrecision(data_loader_val, device)
    }

    evaluator_val = create_supervised_evaluator(model, metrics=metrics, device=device)

    # Logging ----------------------------------------------------------------------------------------------------------
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

        if (epoch_iteration - 1) % config["print_frequency"] == 0:
            output = engine.state.output
            delimiter = "    "

            log_items = list()
            log_items.append(" [{:4d}]".format(epoch_iteration))
            log_items += ["{}: {:.4f}".format(key, output[key]) for key in output]

            log_msg = delimiter.join(log_items)
            print(log_msg)

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
        trainer.run(data_loader_train, max_epochs=config["max_epochs"])
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


def get_data_loaders(data_root, subset_train="training", subset_val="validation", batch_size_train=1, batch_size_val=1):
    class_name_dict = {
        1: "fiber"
    }

    dataset_train = Dataset(data_root,
                            subset_train,
                            transforms=get_transform(),
                            class_name_dict=class_name_dict)
    data_loader_train = DataLoader(dataset_train,
                                   batch_size=batch_size_train,
                                   shuffle=True,
                                   num_workers=4,
                                   collate_fn=collate_fn)

    dataset_val = Dataset(data_root,
                          subset_val,
                          class_name_dict=class_name_dict)
    data_loader_val = DataLoader(dataset_val,
                                 batch_size=batch_size_val,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=collate_fn)

    return data_loader_train, data_loader_val


if __name__ == "__main__":
    main()

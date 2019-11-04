import torch
from torch.utils.data import DataLoader
from torchvision_detection_references.utils import collate_fn
import torchvision_detection_references.transforms as T
from dataset import Dataset
from os import path
from utilities import get_time_stamp, set_random_seed
from mask_rcnn import get_mask_rcnn_model, create_mask_rcnn_trainer
from ignite.engine import create_supervised_evaluator, Events
from metrics import AveragePrecision
from tensorboardX import SummaryWriter


def main():
    # Parameters -------------------------------------------------------------------------------------------------------
    n_classes = 2  # Background and Fiber

    model_name = "mrcnn"

    batch_size_train = 4
    batch_size_val = 1
    max_epochs = 10
    random_seed = 12345
    print_frequency = 10

    subset_train = "training"
    subset_val = "validation"

    # subset_train += "_mini"
    # subset_val += "_mini"

    # Reproducibility --------------------------------------------------------------------------------------------------
    set_random_seed(random_seed)

    # Model ------------------------------------------------------------------------------------------------------------
    model = get_mask_rcnn_model(n_classes)
    model.name = model_name

    # Paths ------------------------------------------------------------------------------------------------------------
    time_stamp = get_time_stamp()
    log_dir = path.join("logs", model.name + "_" + time_stamp)
    data_root = path.join("D:\\", "sciebo", "Dissertation", "Referenzdaten", "IUTA", "easy_images",
                          "individual_fibers_no_clutter_no_loops")

    # Data -------------------------------------------------------------------------------------------------------------
    # TODO: Test pillow-SIMD
    data_loader_train, data_loader_val = \
        get_data_loaders(data_root,
                         subset_train=subset_train, subset_val=subset_val,
                         batch_size_train=batch_size_train, batch_size_val=batch_size_val)

    # Tensorboard ------------------------------------------------------------------------------------------------------
    tensorboard_writer = SummaryWriter(log_dir=log_dir, max_queue=1)

    # Device -----------------------------------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Optimizer --------------------------------------------------------------------------------------------------------
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(parameters, lr=1e-3)

    # Trainer ----------------------------------------------------------------------------------------------------------
    trainer = create_mask_rcnn_trainer(model, optimizer, data_loader_train, device)

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
        print(" Validation:")
        evaluator_val.run(data_loader_val)
        metrics["AP"].print()

        epoch = engine.state.epoch

        tensorboard_writer.add_scalar("training/loss", engine.state.output["loss"], epoch)
        tensorboard_writer.add_scalar("training/lr", engine.state.output["lr"], epoch)
        tensorboard_writer.add_scalar("validation/AP", metrics["AP"].value, epoch)

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
    def increment_(engine):
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

    # Training ---------------------------------------------------------------------------------------------------------
    try:
        trainer.run(data_loader_train, max_epochs=max_epochs)
    finally:
        pass
        tensorboard_writer.close()


def get_transform(train):
    transforms = [T.ToTensor()]

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)


def get_data_loaders(data_root, subset_train="training", subset_val="validation", batch_size_train=1, batch_size_val=1):
    class_name_dict = {
        1: "fiber"
    }

    dataset_train = Dataset(data_root,
                            subset_train,
                            transforms=get_transform(train=True),
                            class_name_dict=class_name_dict)
    data_loader_train = DataLoader(dataset_train,
                                   batch_size=batch_size_train,
                                   shuffle=True,
                                   num_workers=4,
                                   collate_fn=collate_fn)

    dataset_val = Dataset(data_root,
                          subset_val,
                          transforms=get_transform(train=False),
                          class_name_dict=class_name_dict)
    data_loader_val = DataLoader(dataset_val,
                                 batch_size=batch_size_val,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=collate_fn)

    return data_loader_train, data_loader_val


if __name__ == "__main__":
    main()

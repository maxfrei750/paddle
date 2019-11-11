import math
import sys
from ignite.engine import Engine
import torchvision_detection_references.utils as utils
import torch


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

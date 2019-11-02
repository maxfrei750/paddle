import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import math
import sys
from ignite.engine import Engine
import torchvision_detection_references.utils as utils
import torch


def get_mask_rcnn_model(num_classes):
    # Load an instance segmentation model pre-trained on COCO.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # Get the number of input features for the classifier.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Get the number of input features for the mask classifier.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one.
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def create_mask_rcnn_trainer(model, optimizer, data_loader, device=None):
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

        # print("epoch: {} - iteration: {} - loss: {}".format(epoch, iteration, loss_value))
        output_dict = dict()

        for key in loss_dict_reduced:
            output_dict[key] = loss_dict_reduced[key].cpu().detach().numpy().item()

        output_dict["loss"] = loss_value

        output_dict["lr"] = optimizer.param_groups[0]["lr"]

        return output_dict

    return Engine(_update)


def create_mask_rcnn_evaluator(model, metrics=None, device=None):
    metrics = metrics or dict()

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            images, targets = batch
            predictions = model(images)
            return predictions, targets

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
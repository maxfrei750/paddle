import torchvision
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import math
import sys
from ignite.engine import Engine
import torchvision_detection_references.utils as utils
from torch import nn


def get_keypoint_rcnn_model(num_classes, num_keypoints):
    # Load a pre-trained model for classification and return only the features.
    resnet50 = torchvision.models.resnet50(pretrained=True)

    layers = [
        resnet50.conv1,
        resnet50.bn1,
        resnet50.relu,
        resnet50.maxpool,
        resnet50.layer1,
        resnet50.layer2,
        resnet50.layer3,
        resnet50.layer4,
        resnet50.avgpool]

    backbone = nn.Sequential(*layers)

    # KeypointRCNN needs to know the number of
    # output channels in a backbone, so we need to add it here.
    backbone.out_channels = resnet50.inplanes

    # Let's make the RPN generate 5 x 3 anchors per spatial
    # location, with 5 different sizes and 3 different aspect
    # ratios. We have a Tuple[Tuple[int]] because each feature
    # map could potentially have different sizes and
    # aspect ratios.
    # TODO: Adjust ROI sizes.
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    # Let's define what are the feature maps that we will
    # use to perform the region of interest cropping, as well as
    # the size of the crop after rescaling.
    # If your backbone returns a Tensor, featmap_names is expected to
    # be [0]. More generally, the backbone should return an
    # OrderedDict[Tensor], and in featmap_names you can choose which
    # feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    keypoint_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                             output_size=14,
                                                             sampling_ratio=2)
    # Put the pieces together inside a FasterRCNN model.
    model = KeypointRCNN(backbone,
                         num_classes=num_classes,
                         rpn_anchor_generator=anchor_generator,
                         box_roi_pool=roi_pooler,
                         keypoint_roi_pool=keypoint_roi_pooler,
                         num_keypoints=num_keypoints)

    return model


def create_keypoint_rcnn_trainer(model, optimizer, data_loader, device=None):
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

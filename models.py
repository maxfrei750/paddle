import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import KeypointRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch import nn


def get_model(config):
    model_name = config["model"]["model_name"].lower()
    n_classes = config["model"]["n_classes"]

    expected_model_names = ["krcnn", "mrcnn"]
    assert model_name in expected_model_names, \
        f"Unknown modelname '{model_name}'. Expected modelname to be in {expected_model_names}."

    if model_name == "mrcnn":
        model = get_mask_rcnn_model(n_classes)
    elif model_name == "krcnn":
        n_keypoints = config["model"]["n_keypoints"]
        model = get_keypoint_rcnn_model(n_classes, n_keypoints)

    model.name = model_name

    return model


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


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model(config):
    n_classes = config["model"]["n_classes"]
    model = get_mask_rcnn_resnet50_model(n_classes)

    # TODO: Replace with
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(
    #     num_classes=n_classes,
    #     pretrained=True,
    #     trainable_backbone_layers=config["model"]["trainable_backbone_layers"],
    # )

    return model


def get_mask_rcnn_resnet50_model(num_classes, pretrained=True):
    # Load an instance segmentation model pre-trained on COCO.
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    # Get the number of input features for the classifier.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Get the number of input features for the mask classifier.
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one.
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model

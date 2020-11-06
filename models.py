import torchvision


def get_model(num_classes, pretrained_backbone=True, trainable_backbone_layers=3):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        num_classes=num_classes,
        pretrained_backbone=pretrained_backbone,
        trainable_backbone_layers=trainable_backbone_layers,
    )

    return model

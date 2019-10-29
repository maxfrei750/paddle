import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision_detection_references.engine import train_one_epoch, evaluate
from torchvision_detection_references.utils import collate_fn
import torchvision_detection_references.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from dataset import Dataset
from os import path

from visualization import display_detection


def get_instance_segmentation_model(num_classes):
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


def get_transform(train):
    transforms = [T.ToTensor()]

    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
    return T.Compose(transforms)


def get_data_loaders(data_root, batch_size_train=1, batch_size_val=1):
    class_name_dict = {
        1: "fiber"
    }

    dataset_train = Dataset(data_root,
                            "training",
                            transforms=get_transform(train=True),
                            class_name_dict=class_name_dict)
    data_loader_train = DataLoader(dataset_train,
                                   batch_size=batch_size_train,
                                   shuffle=True,
                                   num_workers=4,
                                   collate_fn=collate_fn)

    dataset_val = Dataset(data_root,
                          "validation",
                          transforms=get_transform(train=False),
                          class_name_dict=class_name_dict)
    data_loader_val = DataLoader(dataset_val,
                                 batch_size=batch_size_val,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=collate_fn)

    return data_loader_train, data_loader_val


def main():
    # Parameters -------------------------------------------------------------------------------------------------------
    n_classes = 2  # Background and Fiber

    batch_size_train = 4
    batch_size_val = 1

    n_epochs = 10

    # Model ------------------------------------------------------------------------------------------------------------
    model = get_instance_segmentation_model(n_classes)

    # Paths ------------------------------------------------------------------------------------------------------------
    data_root = path.join("D:\\", "sciebo", "Dissertation", "Referenzdaten", "IUTA", "easy_images",
                          "individual_fibers_no_clutter_no_loops")

    # Data -------------------------------------------------------------------------------------------------------------
    # TODO: Test pillow-SIMD
    data_loader_train, data_loader_val = get_data_loaders(data_root, batch_size_train, batch_size_val)

    # Device -----------------------------------------------------------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Optimizer --------------------------------------------------------------------------------------------------------
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(parameters, lr=1e-3)

    # Learning rate scheduler ------------------------------------------------------------------------------------------
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # Training ---------------------------------------------------------------------------------------------------------
    for epoch in range(n_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_val, device=device)

    # Testing ----------------------------------------------------------------------------------------------------------
    dataset_test = Dataset(data_root,
                           "test",
                           transforms=get_transform(train=False)
                           )

    test_image, _ = dataset_test[0]

    model.eval()
    with torch.no_grad():
        prediction = model([test_image])

    display_detection(test_image, prediction)


if __name__ == "__main__":
    main()

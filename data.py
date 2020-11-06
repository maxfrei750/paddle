from glob import glob
from os import path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, subset, transforms=None, class_name_dict=None):
        assert path.isdir(root), "The specified root does not exist: " + root
        self.root = root

        self.subset_path = path.join(root, subset)
        assert path.isdir(self.subset_path), (
            "The specified subset folder does not exist: " + subset
        )

        self.subset = subset
        self.sample_folders = glob(path.join(root, subset, "**"))

        self.transforms = transforms

        if class_name_dict is None:
            self.class_name_dict = {1: "particle"}
        else:
            self.class_name_dict = class_name_dict

    def __getitem__(self, index):
        sample_folder = self.sample_folders[index]

        image_path = glob(path.join(sample_folder, "images", "*"))[0]
        mask_paths = glob(path.join(sample_folder, "masks", "*"))

        # TODO: Support multiple classes.
        #  instance_class_path = path.join(sample_folder, "*.txt")

        image = Image.open(image_path)
        image = image.convert("RGB")

        n_instances = len(mask_paths)

        masks = list()
        boxes = list()

        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert("1")

            box = list(mask.getbbox())
            boxes.append(box)

            mask = np.array(mask)
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # TODO: Support multiple classes.
        labels = torch.ones((n_instances,), dtype=torch.int64)
        scores = torch.ones((n_instances,), dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Assume that there are no crowd instances.
        iscrowd = torch.zeros((n_instances,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        image = F.to_tensor(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.sample_folders)


def get_data_loaders(data_root, config, transforms=None, collate_fn=None):
    subset_training = config["data"]["subset_training"]
    subset_validation = config["data"]["subset_validation"]
    batch_size_training = config["data"]["batch_size_training"]
    batch_size_validation = config["data"]["batch_size_validation"]
    class_names = config["data"]["class_names"]
    n_data_loader_workers = config["data"]["n_data_loader_workers"]

    dataset_train = Dataset(
        data_root, subset_training, transforms=transforms, class_name_dict=class_names
    )
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size_training,
        shuffle=True,
        num_workers=n_data_loader_workers,
        collate_fn=collate_fn,
    )

    dataset_val = Dataset(data_root, subset_validation, class_name_dict=class_names)
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size_validation,
        shuffle=True,
        num_workers=n_data_loader_workers,
        collate_fn=collate_fn,
    )

    return data_loader_train, data_loader_val


if __name__ == "__main__":
    import random

    import torch.cuda as cuda

    from visualization import display_detection

    test_root_path = "data"

    class_name_dict = {1: "particle"}

    dataset = Dataset(
        test_root_path, subset="validation", class_name_dict=class_name_dict
    )

    sample_id = random.randint(1, len(dataset) - 1)
    image, target = dataset[sample_id]

    if cuda.is_available():
        image = image.to("cuda")

        for key in target:
            target[key] = target[key].to("cuda")

    display_detection(
        image, target, class_name_dict=dataset.class_name_dict, do_display_mask=True
    )

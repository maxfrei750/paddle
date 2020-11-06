from glob import glob
from os import path

import numpy as np
from PIL import Image

import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, subset, transforms=None, class_name_dict=None):
        assert path.isdir(root), "The specified root does not exist: " + root
        self.root = root

        self.subset_path = path.join(root, subset)
        assert path.isdir(self.subset_path), "The specified subset folder does not exist: " + subset

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
        image = np.array(image)

        n_instances = len(mask_paths)

        masks = list()
        boxes = list()

        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert("1")

            box = list(mask.getbbox())
            boxes.append(box)

            mask = np.array(mask)
            masks.append(mask)

        # TODO: Support multiple classes.
        labels = np.ones((n_instances,), dtype=np.int64)

        if self.transforms is not None:
            transformed_data = self.transforms(
                image=image, masks=masks, bboxes=boxes, class_labels=labels
            )

            image = transformed_data["image"]
            masks = transformed_data["masks"]
            boxes = transformed_data["bboxes"]
            labels = transformed_data["class_labels"]

            # Filter empty masks.
            masks = [mask for mask in masks if np.any(mask)]

            n_instances = len(labels)

        labels = torch.as_tensor(labels)
        scores = torch.ones((n_instances,), dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.array(masks, dtype=np.uint8))
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Assume that there are no crowd instances.
        iscrowd = torch.zeros((n_instances,), dtype=torch.uint8)

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

        return image, target

    def __len__(self):
        return len(self.sample_folders)


def get_data_loader(
    data_root, subset, batch_size, class_names, num_workers, transforms=None, collate_fn=None
):
    dataset = Dataset(data_root, subset, transforms=transforms, class_name_dict=class_names)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )

    return data_loader


if __name__ == "__main__":
    import random

    import torch.cuda as cuda

    from visualization import display_detection

    test_root_path = "data"

    class_name_dict = {1: "particle"}

    dataset = Dataset(test_root_path, subset="validation", class_name_dict=class_name_dict)

    sample_id = random.randint(1, len(dataset) - 1)
    image, target = dataset[sample_id]

    if cuda.is_available():
        image = image.to("cuda")

        for key in target:
            target[key] = target[key].to("cuda")

    display_detection(image, target, class_name_dict=dataset.class_name_dict, do_display_mask=True)

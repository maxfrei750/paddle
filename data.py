from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, subset, transforms=None, class_name_dict=None):

        root = Path(root)
        self.root = root
        assert root.is_dir(), "The specified root does not exist: " + root

        self.subset_path = root / subset
        assert self.subset_path.is_dir(), "The specified subset folder does not exist: " + subset

        self.subset = subset

        # For large datasets it might be better to keep the glob-generator.
        self.image_paths = list(self.subset_path.glob("image_*.*"))

        assert self.image_paths, "No images found based on 'image_*.*'."

        self.transforms = transforms

        if class_name_dict is None:
            self.class_name_dict = {1: "particle"}
        else:
            self.class_name_dict = class_name_dict

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        image_name = image_path.stem[6:]

        mask_paths = list(self.subset_path.glob(f"mask_{image_name}*.*"))

        # TODO: Support multiple classes.

        image = Image.open(image_path)
        image = image.convert("RGB")
        image = np.array(image)

        num_instances = len(mask_paths)

        masks = list()

        for mask_path in mask_paths:
            mask = Image.open(mask_path).convert("1")
            mask = np.array(mask)
            masks.append(mask)

        if self.transforms is not None:
            transformed_data = self.transforms(image=image, masks=masks)

            image = transformed_data["image"]
            masks = transformed_data["masks"]

            # Filter empty masks.
            masks = _remove_empty_masks(masks)

            num_instances = len(masks)

        # TODO: Support multiple classes.
        labels = np.ones((num_instances,), dtype=np.int64)
        boxes = extract_bounding_boxes(masks)

        if len(boxes):
            areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            areas = []

        labels = torch.as_tensor(labels)
        scores = torch.ones((num_instances,), dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.tensor(np.array(masks, dtype=np.uint8))
        image_id = torch.tensor([index])
        areas = torch.as_tensor(areas)
        # Assume that there are no crowd instances.
        iscrowd = torch.zeros((num_instances,), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd,
            "image_name": image_name,
        }

        image = F.to_tensor(image)

        return image, target

    def __len__(self):
        return len(self.image_paths)


def extract_bounding_boxes(masks):
    boxes = list()
    for mask in masks:
        box = extract_bounding_box(mask)
        boxes.append(box)

    boxes = np.asarray(boxes)

    return boxes


def extract_bounding_box(mask):
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1]) + 1
    ymin = np.min(pos[0])
    ymax = np.max(pos[0]) + 1
    box = [xmin, ymin, xmax, ymax]
    return box


def _remove_empty_masks(masks):
    return [mask for mask in masks if np.any(mask)]


def get_data_loader(
    data_root, subset, batch_size, class_names, num_workers, transforms=None, collate_fn=None
):
    dataset = Dataset(data_root, subset, transforms=transforms, class_name_dict=class_names)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )

    return data_loader

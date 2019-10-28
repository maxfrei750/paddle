from os import path
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from glob import glob
from visualization import display_detection
import random


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, class_name_dict=None):
        self.root = root
        self.transforms = transforms
        self.sample_folders = glob(path.join(root, "**"))

        if class_name_dict is None:
            self.class_name_dict = {
                1: "particle"
            }
        else:
            self.class_name_dict = class_name_dict

    def __getitem__(self, index):
        sample_folder = self.sample_folders[index]

        image_path = glob(path.join(sample_folder, "images", "*"))[0]
        mask_paths = glob(path.join(sample_folder, "masks", "*"))

        # TODO: Support splines.
        # spline_paths = glob(path.join(sample_folder, "splines", "*"))

        # TODO: Support multiple classes.
        # instance_class_path = path.join(sample_folder, "*.txt")

        image = Image.open(image_path)
        image = image.convert("RGB")

        n_instances = len(mask_paths)

        masks = list()
        boxes = list()

        for mask_path in mask_paths:
            mask = Image.open(mask_path)

            box = mask.getbbox()
            boxes.append(box)

            mask = np.array(mask)
            masks.append(mask)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # TODO: Support multiple classes.
        labels = torch.ones((n_instances,), dtype=torch.int64)

        scores = torch.ones((n_instances,), dtype=torch.float32)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([index])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Assume that there are no crowd instances.
        is_crowd = torch.zeros((n_instances,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "is_crowd": is_crowd}

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.sample_folders)


if __name__ == '__main__':
    test_root_path = path.join("D:\\", "sciebo", "Dissertation", "Referenzdaten", "IUTA", "easy_images",
                               "individual_fibers_no_clutter_no_loops", "training")

    class_name_dict = {
        1: "fiber"
    }

    dataset = Dataset(test_root_path, class_name_dict=class_name_dict)

    sample_id = random.randint(1, len(dataset))
    image, target = dataset[sample_id]

    display_detection(image,
                      target,
                      class_name_dict=dataset.class_name_dict)

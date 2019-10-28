from os import path
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from glob import glob
from visualization import display_detection


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.sample_folders = glob(path.join(root, "**"))

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

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([index])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Assume that there are no crowd instances.
        is_crowd = torch.zeros((n_instances,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
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

    dataset = Dataset(test_root_path)
    image, target = dataset.__getitem__(3)

    display_detection(image, target)

from os import path
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from glob import glob
from torchvision.transforms import functional as F
import pandas as pd
from torch.utils.data import DataLoader


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
            self.class_name_dict = {
                1: "particle"
            }
        else:
            self.class_name_dict = class_name_dict

    def __getitem__(self, index):
        sample_folder = self.sample_folders[index]

        image_path = glob(path.join(sample_folder, "images", "*"))[0]
        mask_paths = glob(path.join(sample_folder, "masks", "*"))
        spline_paths = glob(path.join(sample_folder, "splines", "*.csv"))

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

        key_point_sets = list()
        spline_widths = list()

        for spline_path in spline_paths:
            spline_data = pd.read_csv(spline_path)

            spline_width = spline_data["width"][0]
            spline_widths.append(spline_width)

            key_point_set = spline_data[["x", "y"]]
            # Assume that all keypoints are visible.
            key_point_set["visibility"] = 1
            key_point_sets.append(key_point_set.to_numpy())

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # TODO: Support multiple classes.
        labels = torch.ones((n_instances,), dtype=torch.int64)
        scores = torch.ones((n_instances,), dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # Assume that there are no crowd instances.
        iscrowd = torch.zeros((n_instances,), dtype=torch.int64)
        key_point_sets = torch.as_tensor(key_point_sets, dtype=torch.float32)
        spline_widths = torch.as_tensor(spline_widths, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "scores": scores,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "keypoints": key_point_sets,
            "spline_widths": spline_widths
        }

        image = F.to_tensor(image)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.sample_folders)


def get_data_loaders(data_root, config, transforms=None, collate_fn=None):
    subset_train = config["subset_train"]
    subset_val = config["subset_val"]
    batch_size_train = config["batch_size_train"]
    batch_size_val = config["batch_size_val"]
    class_names = config["class_names"]
    n_data_loader_workers = config["n_data_loader_workers"]

    dataset_train = Dataset(data_root,
                            subset_train,
                            transforms=transforms,
                            class_name_dict=class_names)
    data_loader_train = DataLoader(dataset_train,
                                   batch_size=batch_size_train,
                                   shuffle=True,
                                   num_workers=n_data_loader_workers,
                                   collate_fn=collate_fn)

    dataset_val = Dataset(data_root,
                          subset_val,
                          class_name_dict=class_names)
    data_loader_val = DataLoader(dataset_val,
                                 batch_size=batch_size_val,
                                 shuffle=True,
                                 num_workers=n_data_loader_workers,
                                 collate_fn=collate_fn)

    return data_loader_train, data_loader_val


if __name__ == '__main__':
    from visualization import display_detection
    import random
    import torch.cuda as cuda

    test_root_path = path.join("D:\\", "sciebo", "Dissertation", "Referenzdaten", "IUTA", "easy_images",
                               "individual_fibers_no_clutter_no_loops")

    class_name_dict = {
        1: "fiber"
    }


    dataset = Dataset(test_root_path,
                      subset="test",
                      class_name_dict=class_name_dict)

    sample_id = random.randint(1, len(dataset))
    image, target = dataset[sample_id]

    if cuda.is_available():
        image = image.to("cuda")

        for key in target:
            target[key] = target[key].to("cuda")

    display_detection(image,
                      target,
                      class_name_dict=dataset.class_name_dict,
                      do_display_mask=False)

import random

from torchvision.transforms import functional as F


def _flip_keypoints_horizontal(key_points, width):
    key_points[:, :, 0] = width - key_points[:, :, 0]
    return key_points


def _flip_keypoints_vertical(key_points, height):
    key_points[:, :, 1] = height - key_points[:, :, 1]
    return key_points


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-2)
            bbox = target["boxes"]
            bbox[:, [1, 3]] = height - bbox[:, [3, 1]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-2)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_keypoints_vertical(keypoints, height)
                target["keypoints"] = keypoints
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_keypoints_horizontal(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

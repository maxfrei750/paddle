import albumentations


def get_transform(training=False):
    transforms = [albumentations.RandomCrop(width=1024, height=1024)]

    if training:
        transforms += [
            albumentations.HorizontalFlip(0.5),
            albumentations.VerticalFlip(0.5),
            albumentations.RandomRotate90(0.75),
        ]

    return albumentations.Compose(transforms)

import albumentations


def get_transform():
    transform = albumentations.Compose(
        [
            albumentations.RandomCrop(width=1024, height=1024),
            albumentations.HorizontalFlip(0.5),
            albumentations.VerticalFlip(0.5),
            albumentations.RandomRotate90(0.75),
        ]
    )

    return transform

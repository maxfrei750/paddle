import albumentations


def get_transform(training=False, cropping_rectangle=None):
    transforms = []

    if cropping_rectangle:
        transforms.append(albumentations.Crop(*cropping_rectangle))

    if training:
        transforms += [
            albumentations.HorizontalFlip(0.5),
            albumentations.VerticalFlip(0.5),
            albumentations.RandomRotate90(0.75),
        ]

    return albumentations.Compose(transforms)

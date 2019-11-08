from torchvision_detection_references.transforms import _flip_keypoints_horizontal, _flip_keypoints_vertical
import numpy as np
from visualization import display_detection


def main():
    np.random.seed(1)
    image = np.zeros((1000, 1000, 3))

    height, width, _ = image.shape

    keypoints = np.random.randint(0, 1000, size=(1, 10, 3))
    keypoints[:, 2] = 1

    detection = {
        "keypoints": [keypoints]
    }

    display_detection(image, detection)

    keypoints = _flip_keypoints_horizontal(keypoints, width)
    keypoints = _flip_keypoints_vertical(keypoints, width)
    display_detection(image, detection)


if __name__ == "__main__":
    main()

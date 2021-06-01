from pathlib import Path

import fire

# from paddle.spline import to_mask
from paddle.data import MaskRCNNDataset


def convert_splines_to_masks(data_root, subset):
    data_root = Path(data_root)
    dataset = MaskRCNNDataset(data_root, subset, num_slices_per_axis=2)
    image, target = dataset[0]

    pass


if __name__ == "__main__":
    convert_splines_to_masks("data/IUTA/easy_images/", "real")

    # fire.Fire(convert_splines_to_masks)

import random

import fire

from data import MaskRCNNDataset
from utilities import AnyPath
from visualization import display_detection


def inspect_data(data_root: AnyPath, subset: str = "training"):
    """Inspect a dataset.

    :param data_root: Root folder of the dataset.
    :param subset: Subset of the dataset. Default: "training"
    """

    dataset = MaskRCNNDataset(data_root, subset=subset)

    sample_id = random.randint(0, len(dataset) - 1)
    image, target = dataset[sample_id]

    display_detection(
        image,
        target,
        map_label_to_class_name=dataset.map_label_to_class_name,
        do_display_box=False,
        do_display_label=False,
        do_display_score=False,
    )


if __name__ == "__main__":
    fire.Fire(inspect_data)

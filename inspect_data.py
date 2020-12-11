import os
import random
import typing

import fire

from data import Dataset
from visualization import display_detection


def inspect_data(data_root: typing.Union[str, bytes, os.PathLike], subset: str = "training"):
    """Inspect a dataset.

    :param data_root: Root folder of the dataset.
    :param subset: Subset of the dataset. Default: "training"
    """

    dataset = Dataset(data_root, subset=subset)

    sample_id = random.randint(1, len(dataset) - 1)
    image, target = dataset[sample_id]

    display_detection(
        image,
        target,
        class_name_dict=dataset.class_name_dict,
        do_display_box=False,
        do_display_label=False,
        do_display_score=False,
    )


if __name__ == "__main__":
    fire.Fire(inspect_data)

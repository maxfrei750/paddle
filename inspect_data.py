import random

import fire

from paddle.data import MaskRCNNDataset
from paddle.utilities import AnyPath
from paddle.visualization import visualize_detection


def inspect_data(data_root: AnyPath, subset: str = "training"):
    """Inspect a dataset.

    :param data_root: Root folder of the dataset.
    :param subset: Subset of the dataset. Default: "training"
    """

    dataset = MaskRCNNDataset(data_root, subset=subset)

    sample_id = random.randint(0, len(dataset) - 1)
    image, target = dataset[sample_id]

    result = visualize_detection(
        image,
        target,
        do_display_box=True,
        do_display_label=False,
        do_display_score=False,
        map_label_to_class_name=dataset.map_label_to_class_name,
        line_width=3,
    )

    result.show()


if __name__ == "__main__":
    fire.Fire(inspect_data)

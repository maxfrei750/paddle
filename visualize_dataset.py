from pathlib import Path
from typing import List, Optional, Tuple

import fire
from tqdm import tqdm

from custom_types import Annotation, AnyPath, Batch, Image
from data import MaskRCNNDataset
from visualization import visualize_detection


def visualize_dataset(
    data_root: AnyPath = "data",
    subset: str = "test",
    output_root: Optional[AnyPath] = None,
):
    """Visualize a dataset.

    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use for the validation.
    :param output_root: Path, where visualizations are saved to. If None, then the visualizations
        are saved into the data subset folder, i.e. data_root/subset.
    """

    if output_root is None:
        output_root = Path(data_root) / subset
    else:
        Path(output_root).mkdir(exist_ok=True, parents=True)

    data_set = MaskRCNNDataset(data_root, subset=subset)

    for image, target in tqdm(data_set):
        image_name = target["image_name"]

        # TODO: Add visualize_detection parameters to visualize_dataset parameters.
        result = visualize_detection(
            image,
            target,
            do_display_box=False,
            do_display_label=False,
            do_display_score=True,
            line_width=3,
        )

        visualization_file_path = output_root / f"visualization_{image_name}.png"
        result.save(visualization_file_path)


if __name__ == "__main__":
    fire.Fire(visualize_dataset)

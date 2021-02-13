from pathlib import Path
from typing import Optional

import fire

from paddle.custom_types import AnyPath
from paddle.postprocessing import Postprocessor, SaveVisualization

# TODO: Add SaveVisualization keywords as arguments.


def visualize_dataset(
    data_root: AnyPath,
    subset: str,
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

    post_processing_steps = [
        SaveVisualization(
            output_root=output_root,
            do_display_box=False,
            do_display_score=False,
            do_display_label=True,
            line_width=2,
        ),
    ]

    Postprocessor(
        data_root, subset, post_processing_steps, progress_bar_description="Visualization: "
    ).run()


if __name__ == "__main__":
    fire.Fire(visualize_dataset)

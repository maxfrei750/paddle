from pathlib import Path
from typing import Dict, Optional

import fire

from paddle.custom_types import AnyPath
from paddle.postprocessing import Postprocessor, SaveVisualization


def visualize_dataset(
    data_root: AnyPath,
    subset: str,
    output_root: Optional[AnyPath] = None,
    file_name_prefix: str = "visualization",
    do_display_box: Optional[bool] = True,
    do_display_label: Optional[bool] = True,
    do_display_score: Optional[bool] = True,
    do_display_mask: Optional[bool] = True,
    do_display_outlines_only: Optional[bool] = True,
    map_label_to_class_name: Optional[Dict[int, str]] = None,
    line_width: Optional[int] = 3,
    font_size: Optional[int] = 16,
):
    """Visualize a dataset.

    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use for the validation.
    :param output_root: Path, where visualizations are saved to. If None, then the visualizations
        are saved into the data subset folder, i.e. data_root/subset.
    :param file_name_prefix: Prefix for visualization output files.
    :param do_display_box: If true and available, then bounding boxes are displayed.
    :param do_display_label: If true and available, then labels are displayed.
    :param do_display_score: If true and available, then scores are displayed.
    :param do_display_mask: If true and available, then masks are displayed.
    :param do_display_outlines_only: If true, only the outlines of masks are displayed.
    :param map_label_to_class_name: Dictionary, which maps instance labels to class names.
    :param line_width: Line width for bounding boxes and mask outlines.
    :param font_size: Font Size for labels and scores.
    """

    if output_root is None:
        output_root = Path(data_root) / subset
    else:
        Path(output_root).mkdir(exist_ok=True, parents=True)

    post_processing_steps = [
        SaveVisualization(
            output_root=output_root,
            file_name_prefix=file_name_prefix,
            do_display_box=do_display_box,
            do_display_label=do_display_label,
            do_display_score=do_display_score,
            do_display_mask=do_display_mask,
            do_display_outlines_only=do_display_outlines_only,
            map_label_to_class_name=map_label_to_class_name,
            line_width=line_width,
            font_size=font_size,
        ),
    ]

    Postprocessor(
        data_root, subset, post_processing_steps, progress_bar_description="Visualization: "
    ).run()


if __name__ == "__main__":
    fire.Fire(visualize_dataset)

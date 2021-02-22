from pathlib import Path
from typing import List, Optional

from .custom_types import AnyPath
from .data import MaskRCNNDataset
from .deployment import run_model_on_dataset
from .lightning_modules import LightningMaskRCNN
from .postprocessing import Postprocessor, SaveVisualization
from .utilities import get_best_checkpoint_path, get_latest_log_folder_path


# TODO: Pass a dataset.
def inspect_dataset(
    data_root: AnyPath,
    subset: str,
    initial_cropping_rectangle: Optional[List[int]] = None,
    output_root: Optional[AnyPath] = None,
    file_name_prefix: str = "visualization",
    do_display_box: Optional[bool] = True,
    do_display_label: Optional[bool] = True,
    do_display_score: Optional[bool] = True,
    do_display_mask: Optional[bool] = True,
    do_display_outlines_only: Optional[bool] = True,
    line_width: Optional[int] = 3,
    font_size: Optional[int] = 16,
):
    """Visualize a dataset.

    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use for the validation.
    :param initial_cropping_rectangle: If not None, [x0, y0, x1, y1] rectangle used for the cropping
        of images.
    :param output_root: Path, where visualizations are saved to. If None, then the visualizations
        are saved into the data subset folder, i.e. data_root/subset.
    :param file_name_prefix: Prefix for visualization output files.
    :param do_display_box: If true and available, then bounding boxes are displayed.
    :param do_display_label: If true and available, then labels are displayed.
    :param do_display_score: If true and available, then scores are displayed.
    :param do_display_mask: If true and available, then masks are displayed.
    :param do_display_outlines_only: If true, only the outlines of masks are displayed.
    :param line_width: Line width for bounding boxes and mask outlines.
    :param font_size: Font Size for labels and scores.
    """

    if output_root is None:
        output_root = Path(data_root) / subset
    else:
        Path(output_root).mkdir(exist_ok=True, parents=True)

    data_set = MaskRCNNDataset(
        data_root, subset, initial_cropping_rectangle=initial_cropping_rectangle
    )

    post_processing_steps = [
        SaveVisualization(
            output_root=output_root,
            file_name_prefix=file_name_prefix,
            do_display_box=do_display_box,
            do_display_label=do_display_label,
            do_display_score=do_display_score,
            do_display_mask=do_display_mask,
            do_display_outlines_only=do_display_outlines_only,
            map_label_to_class_name=data_set.map_label_to_class_name,
            line_width=line_width,
            font_size=font_size,
        ),
    ]

    Postprocessor(data_set, post_processing_steps, progress_bar_description="Visualization: ").run()


# TODO: Pass a dataset.
def inspect_model(
    log_root: AnyPath,
    data_root: AnyPath,
    subset: str,
    model_id: Optional[str] = None,
    initial_cropping_rectangle: Optional[List[int]] = None,
):
    """Inspect a model by applying it to a validation dataset and store the results in the model
        folder.

    :param model_id: Identifier of the model or None. If None, then the latest trained model is
        inspected.
    :param log_root: Path of the log folder.
    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use for the inspection.
    :param initial_cropping_rectangle: If not None, [x0, y0, x1, y1] rectangle used for the cropping
        of images.
    """

    if model_id is None:
        model_id = get_latest_log_folder_path(log_root)

    model_root = Path(log_root) / model_id
    result_root = model_root / "results"
    checkpoint_root = model_root / "checkpoints"

    checkpoint_path = get_best_checkpoint_path(checkpoint_root)

    model = LightningMaskRCNN.load_from_checkpoint(checkpoint_path)

    run_model_on_dataset(model, result_root, data_root, subset, initial_cropping_rectangle)

    inspect_dataset(
        result_root,
        subset,
    )

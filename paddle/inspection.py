from pathlib import Path
from typing import Dict, List, Optional

from pytorch_lightning import Trainer

from .callbacks import TestPredictionWriter
from .custom_types import AnyPath
from .data import MaskRCNNDataModule
from .models import LightningMaskRCNN
from .postprocessing import Postprocessor, SaveVisualization
from .utilities import get_best_checkpoint_path, get_latest_log_folder_path


def inspect_dataset(
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


def inspect_model(
    log_root: AnyPath,
    data_root: AnyPath,
    subset: str,
    model_id: Optional[str] = None,
    cropping_rectangle: Optional[List[int]] = None,
):
    """Inspect a model by applying it to a validation dataset and store the results in the model
        folder.

    :param model_id: Identifier of the model or None. If None, then the latest trained model is
        inspected.
    :param log_root: Path of the log folder.
    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use for the validation.
    :param cropping_rectangle: If not None, rectangle to use for the cropping of the validation
        images.
    """

    if model_id is None:
        model_id = get_latest_log_folder_path(log_root)

    model_root = Path(log_root) / model_id
    result_root = model_root / "results"
    checkpoint_root = model_root / "checkpoints"

    checkpoint_path = get_best_checkpoint_path(checkpoint_root)
    model = LightningMaskRCNN.load_from_checkpoint(checkpoint_path)

    data_module = MaskRCNNDataModule(
        data_root, cropping_rectangle=cropping_rectangle, test_subset=subset
    )
    data_module.setup()

    map_label_to_class_name = data_module.test_dataset.map_label_to_class_name

    trainer = Trainer(
        logger=False,
        checkpoint_callback=False,
        gpus=-1,
        callbacks=[
            TestPredictionWriter(
                result_root / subset, map_label_to_class_name=map_label_to_class_name
            )
        ],
    )

    trainer.test(model=model, datamodule=data_module, verbose=False)

    inspect_dataset(
        result_root,
        subset,
        map_label_to_class_name=map_label_to_class_name,
    )

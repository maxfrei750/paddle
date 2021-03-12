from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer

from .callbacks.test import ConfusionMatrixSaver, PredictionWriter
from .custom_types import AnyPath, CroppingRectangle
from .data import MaskRCNNDataModule, MaskRCNNDataset
from .lightning_modules import LightningMaskRCNN
from .postprocessing import (
    FilterBorderInstances,
    Postprocessor,
    SaveMaskProperties,
    SaveVisualization,
    calculate_maximum_feret_diameters,
)
from .visualization import plot_particle_size_distributions


def run_model_on_dataset(
    model: LightningMaskRCNN,
    output_root: AnyPath,
    data_root: AnyPath,
    subset: str,
    initial_cropping_rectangle: Optional[CroppingRectangle] = None,
    num_slices_per_axis: Optional[int] = 1,
) -> None:
    """Loads a model, runs a dataset through it and stores the results.

    :param model: Model to use.
    :param output_root: Path where detections are saved.
    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use.
    :param initial_cropping_rectangle: If not None, [x_min, y_min, x_max, y_max] rectangle used for
        the cropping of images.
    :param num_slices_per_axis: Integer number of slices per image axis. `num_slices_per_axis`=n
        will result in n² pieces. Slicing is performed after the initial cropping and before the
        user transform.
    """

    output_root = Path(output_root) / subset

    data_module = MaskRCNNDataModule(
        data_root,
        initial_cropping_rectangle=initial_cropping_rectangle,
        test_subset=subset,
        num_slices_per_axis=num_slices_per_axis,
    )
    data_module.setup()

    trainer = Trainer(
        logger=False,
        checkpoint_callback=False,
        gpus=-1,
        callbacks=[
            PredictionWriter(
                output_root,
                map_label_to_class_name=data_module.test_dataset.map_label_to_class_name,
            ),
            ConfusionMatrixSaver(
                output_root,
                class_names=data_module.test_dataset.class_names,
                iou_type="mask",
                iou_threshold=0.5,
            ),
        ],
    )
    trainer.test(model=model, datamodule=data_module, verbose=False)


def default_analysis(
    model_checkpoint_path: AnyPath,
    output_root: AnyPath,
    data_root: AnyPath,
    subset: str,
    initial_cropping_rectangle: Optional[CroppingRectangle] = None,
    num_slices_per_axis: Optional[int] = 1,
) -> None:
    """Performs a default analysis of a dataset using a given model. The analysis includes the
        filtering of border instances, a visualization and the measurement of the area equivalent
        diameter, as well as the minimum and maximum Feret diameters.

    :param model_checkpoint_path: Path of the model checkpoint to load and use.
    :param output_root: Path where detections are saved.
    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use.
    :param initial_cropping_rectangle: If not None, [x_min, y_min, x_max, y_max] rectangle used for
        the cropping of images.
    :param num_slices_per_axis: Integer number of slices per image axis. `num_slices_per_axis`=n
        will result in n² pieces. Slicing is performed after the initial cropping and before the
        user transform.
    """
    output_root = Path(output_root)
    output_path = output_root / subset

    model_checkpoint_path = Path(model_checkpoint_path)

    model = LightningMaskRCNN.load_from_checkpoint(model_checkpoint_path)

    run_model_on_dataset(
        model,
        output_root,
        data_root,
        subset,
        initial_cropping_rectangle=initial_cropping_rectangle,
        num_slices_per_axis=num_slices_per_axis,
    )

    result_data_set = MaskRCNNDataset(
        output_root,
        subset=subset,
    )

    measurement_csv_path = output_path / "measurements.csv"
    measurement_fcns = {
        "feret_diameter_max": calculate_maximum_feret_diameters,
    }
    post_processing_steps = [
        FilterBorderInstances(border_width=3),
        SaveMaskProperties(
            measurement_csv_path,
            measurement_fcns=measurement_fcns,
        ),
        SaveVisualization(
            output_root=output_path,
            do_display_box=False,
            do_display_score=False,
            do_display_label=False,
            line_width=2,
            map_label_to_class_name=result_data_set.map_label_to_class_name,
        ),
    ]

    postprocessor = Postprocessor(result_data_set, post_processing_steps)
    postprocessor.log(output_path)
    postprocessor.run()

    measurement_results = pd.read_csv(measurement_csv_path, dtype={"image_name": str})

    plot_particle_size_distributions(
        particle_size_lists=[measurement_results["feret_diameter_max"]],
        score_lists=[measurement_results["score"]],
        measurand_name="Maximum Feret Diameter",
        labels=[subset],
    )

    plt.savefig(output_path / "particle_size_distribution.pdf")

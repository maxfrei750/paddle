from pathlib import Path
from typing import Optional

import pandas as pd
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer

from .callbacks import TestPredictionWriter
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
    cropping_rectangle: Optional[CroppingRectangle] = None,
) -> None:
    """Loads a model, runs a dataset through it and stores the results.

    :param model: Model to use.
    :param output_root: Path where detections are saved.
    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use.
    :param cropping_rectangle: If not None, [x0, y0, x1, y1] rectangle used for the cropping of
        images. Applied before all other transforms.
    """

    output_root = Path(output_root)

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
                output_root / subset, map_label_to_class_name=map_label_to_class_name
            )
        ],
    )
    trainer.test(model=model, datamodule=data_module, verbose=False)


def default_analysis(
    model_checkpoint_path: AnyPath,
    output_root: AnyPath,
    data_root: AnyPath,
    subset: str,
    cropping_rectangle: Optional[CroppingRectangle] = None,
) -> None:
    """Performs a default analysis of a dataset using a given model. The analysis includes the
        filtering of border instances, a visualization and the measurement of the area equivalent
        diameter, as well as the minimum and maximum Feret diameters.

    :param model_checkpoint_path: Path of the model checkpoint to load and use.
    :param output_root: Path where detections are saved.
    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use.
    :param cropping_rectangle: If not None, [x0, y0, x1, y1] rectangle used for the cropping of
        images. Applied before all other transforms.
    """
    output_root = Path(output_root)
    output_path = output_root / subset

    model_checkpoint_path = Path(model_checkpoint_path)

    model = LightningMaskRCNN.load_from_checkpoint(model_checkpoint_path)

    run_model_on_dataset(model, output_root, data_root, subset, cropping_rectangle)

    result_data_set = MaskRCNNDataset(
        output_root,
        subset=subset,
        cropping_rectangle=cropping_rectangle,
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

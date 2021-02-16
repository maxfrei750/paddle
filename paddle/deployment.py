from pathlib import Path
from typing import Optional

from pytorch_lightning import Trainer

from .callbacks import TestPredictionWriter
from .custom_types import AnyPath, CroppingRectangle
from .data import MaskRCNNDataModule
from .models import LightningMaskRCNN


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

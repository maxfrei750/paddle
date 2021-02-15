from pathlib import Path
from typing import List, Optional

import fire
from pytorch_lightning import Trainer

from paddle.callbacks import TestPredictionWriter
from paddle.custom_types import AnyPath
from paddle.data import MaskRCNNDataModule
from paddle.models import LightningMaskRCNN
from paddle.utilities import get_best_checkpoint_path, get_latest_log_folder_path
from paddle.visualization import visualize_dataset


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

    visualize_dataset(
        result_root,
        subset,
        map_label_to_class_name=map_label_to_class_name,
    )


if __name__ == "__main__":
    fire.Fire(inspect_model)

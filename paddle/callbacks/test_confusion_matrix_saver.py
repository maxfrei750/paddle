from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
from pytorch_lightning import LightningModule, Trainer, callbacks

from ..custom_types import AnyPath, Batch, TestOutput
from ..metrics import ConfusionMatrix


class TestConfusionMatrixSaver(callbacks.Callback):
    """Saves a confusion matrix.

    :param output_root: Root folder for the output.
    :param class_names: List of class names to be used as labels for the confusion matrix.
    :param iou_type: Type of Intersection over Union (IOU) used to determine if a prediction matches
        a target. Either "box" or "mask".
    :param iou_threshold: IOU threshold, above which a prediction is considered a match for a
        target.
    :param score_threshold: Score threshold, above which a prediction is considered non-background.
        default: 0.5
    :param normalize: Normalization mode for confusion matrix. Choose from
            - ``None``: no normalization (default)
            - ``'true'``: normalization over the targets (most commonly used)
            - ``'pred'``: normalization over the predictions
            - ``'all'``: normalization over the whole matrix
    """

    def __init__(
        self,
        output_root: AnyPath,
        class_names: List[str],
        iou_type: Literal["box", "mask"],
        iou_threshold: float,
        score_threshold: float = 0.5,
        normalize: Optional[str] = None,
        file_name: str = "confusion_matrix.pdf",
    ) -> None:
        """"""
        super().__init__()

        self.class_names = class_names
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.file_name = file_name

        self.confusion_matrix = ConfusionMatrix(
            len(class_names), iou_type, iou_threshold, score_threshold, normalize
        )

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: TestOutput,
        batch: Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Update the confusion matrix.

        :param trainer: Lightning Trainer
        :param pl_module: Lightning Module
        :param outputs: Outputs of the current test batch.
        :param batch: Current batch.
        :param batch_idx: Current batch id.
        :param dataloader_idx: Dataloader id.
        """
        _, targets = batch

        predictions = outputs["predictions"]

        self.confusion_matrix.update(predictions, targets)

    def on_test_end(self, trainer, pl_module):
        """Save the confusion matrix at the end of the test.

        :param trainer: Lightning Trainer
        :param pl_module: Lightning Module
        """
        output_path = self.output_root / self.file_name
        figure = self.confusion_matrix.plot(self.class_names)
        figure.savefig(output_path)
        plt.close(figure)

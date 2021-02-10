from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from custom_types import AnyPath
from data import MaskRCNNDataLoader, MaskRCNNDataset

from .postprocessingsteps import PostProcessingStepBase


class Postprocessor:
    """Postprocess a dataset, by applying a list of post processing steps.

    :param data_root: Path of the data folder.
    :param subset: Name of the subset to use for the validation.
    :param post_processing_steps: List of post processing steps.
    """

    def __init__(
        self,
        data_root: AnyPath,
        subset: str,
        post_processing_steps: List[PostProcessingStepBase],
    ) -> None:
        self.data_root = data_root
        self.data_set = MaskRCNNDataset(data_root, subset)
        self.data_loader = MaskRCNNDataLoader(self.data_set, batch_size=1, shuffle=False)
        self.data_iterator = iter(self.data_loader)

        self.post_processing_steps = post_processing_steps

    def run(self):
        """Iterate all samples of a dataset and apply postprocessing steps to each sample.

        :return:
        """
        for images, targets in tqdm(self.data_iterator, desc="Postprocessing: "):
            for image, target in zip(images, targets):
                for post_processing_step in self.post_processing_steps:
                    image, target = post_processing_step(image, target)

    def __str__(self) -> str:
        """The string representation of this class is a summary of the post processing steps in yaml
        format."""
        return "".join(
            [str(post_processing_step) for post_processing_step in self.post_processing_steps]
        )

    def log(self, output_root, output_file_name: Optional[str] = "postprocessing_log.yaml") -> None:
        """Write a summary of the postprocessing steps to disk.

        :param output_root: Folder where the log is saved.
        :param output_file_name: Name of the log file.
        """
        log_file_path = Path(output_root) / output_file_name
        with open(log_file_path, "w") as file:
            file.write(str(self))

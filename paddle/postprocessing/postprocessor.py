from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from ..data import MaskRCNNDataLoader, MaskRCNNDataset
from .postprocessingsteps import Numpify


class Postprocessor:
    """Postprocess a dataset, by applying a list of post processing steps.

    :param data_set: Data set to postprocess.
    :param post_processing_steps: List of post processing steps.
    :param progress_bar_description: Description of the tqdm progress bar.
    """

    def __init__(
        self,
        data_set: MaskRCNNDataset,
        post_processing_steps: List,
        progress_bar_description: Optional[str] = "Postprocessing: ",
    ) -> None:
        self.data_set = data_set
        self.data_loader = MaskRCNNDataLoader(self.data_set, batch_size=1, shuffle=False)
        self.data_iterator = iter(self.data_loader)

        self.post_processing_steps = [Numpify()] + post_processing_steps

        self.progress_bar_description = progress_bar_description

    def run(self):
        """Iterate all samples of a dataset and apply postprocessing steps to each sample.

        :return:
        """
        for images, targets in tqdm(self.data_iterator, desc=self.progress_bar_description):
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

        if log_file_path.exists():
            raise FileExistsError(f"File already exists: {log_file_path}")

        with open(log_file_path, "w") as file:
            file.write(str(self))

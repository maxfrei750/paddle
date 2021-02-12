from typing import Any, Dict, Optional

from pytorch_lightning import callbacks


class ModelCheckpoint(callbacks.ModelCheckpoint):
    """Replaces slashes in the checkpoint filename with underscores to prevent the unwanted
    creation of directories."""

    # TODO: Replace with pytorch_lightning.callbacks.ModelCheckpoint as soon as
    #  https://github.com/PyTorchLightning/pytorch-lightning/issues/4012 is resolved.

    @classmethod
    def _format_checkpoint_name(
        cls,
        filename: Optional[str],
        epoch: int,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> str:
        filename = super()._format_checkpoint_name(filename, epoch, step, metrics, prefix)
        filename = filename.replace("/", "_")
        return filename

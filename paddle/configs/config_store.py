from hydra.core.config_store import ConfigStore

from .rcnns import MaskRCNNConfig


def populate() -> None:
    """Register configs."""
    cs = ConfigStore.instance()

    cs.store(name="MaskRCNN", node=MaskRCNNConfig, provider="paddle")
    # cs.store(name="FibeRCNN", node=FibeRCNNConfig, provider="paddle")

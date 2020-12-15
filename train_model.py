from pathlib import Path

import fire
from config import Config
from training import training


def train_model(config_name="maskrcnn"):
    """Trains a default model.

    :param config_name: File name of a configuration file (without yml-extension).
    """
    config_file_name = config_name + ".yml"
    config_path = Path("configs") / config_file_name
    config = Config.load(config_path)
    config["name"] = config_name

    training(config)


if __name__ == "__main__":
    fire.Fire(train_model)

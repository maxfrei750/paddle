from os import path

from config import Config
from training import training


def main(config_name="maskrcnn"):
    config = Config.load(path.join("configs", config_name + ".yml"))
    config["name"] = config_name

    training(config)


if __name__ == "__main__":
    main()

from collections import UserDict

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class Config(UserDict):
    @staticmethod
    def load(path):
        with open(path) as file:
            return Config(yaml.load(file, Loader=Loader))

    def save(self, path):
        with open(path, "w") as file:
            yaml.dump(self.data, file)


def main():
    import os
    config = Config.load(os.path.join("configs", "mrcnn.yml"))
    print(config)
    print(config["model"]["model_name"])
    print(config["teasdasfkljdslgas"])
    print(config["data"]["class_names"][1])


if __name__ == "__main__":
    main()

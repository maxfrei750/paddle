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

    def __getitem__(self, key):
        if key not in self:
            return None
        else:
            return super().__getitem__(key)


def main():
    import os
    config = Config.load(os.path.join("configs", "mrcnn.yml"))
    print(config)
    print(config["model_name"])
    print(config["teasdasfkljdslgas"])


if __name__ == "__main__":
    main()

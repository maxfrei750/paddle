from pathlib import Path

import torch
from config import Config
from ignite.handlers import Checkpoint
from models import get_model


def load_trained_model(model_path, device):
    model_path = Path(model_path)
    config_path = list(model_path.parent.glob("*.yml"))[0]
    config = Config.load(config_path)
    model = get_model(config["model"]["n_classes"])
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
    model.eval()
    return model

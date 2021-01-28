from pathlib import Path

import torch
from config import Config
from ignite.handlers import Checkpoint

from models import get_model

# TODO: Fix.
# TODO: Add typehints
# TODO: Add docstrings


def load_trained_model(model_path, device):
    model_path = Path(model_path)
    config_path = list(model_path.parent.glob("*.yml"))[0]
    config = Config.load(config_path)
    model = get_model(config["model"]["n_classes"], config["model"]["n_detections_per_image_max"])
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
    model.eval()
    return model


def analyze_image(model, image):
    image = image.squeeze()

    device = next(model.parameters()).device
    with torch.no_grad():
        prediction = model([image.to(device)])[0]

    for measurand in ["scores", "labels", "boxes"]:
        prediction[measurand] = list(prediction[measurand].cpu().numpy())

    prediction["masks"] = list(prediction["masks"].squeeze().round().cpu().numpy().astype("bool"))

    return prediction

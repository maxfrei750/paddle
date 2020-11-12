from pathlib import Path
from statistics import gmean, gstd

import matplotlib.pyplot as plt
import numpy as np
import yaml

import torch
from config import Config
from data import Dataset
from ignite.handlers import Checkpoint
from models import get_model
from postprocessing import filter_border_particles
from torch.utils.data import DataLoader
from transforms import get_transform
from utilities import get_best_model_path
from visualization import get_viridis_colors, save_visualization


def inspect_model():
    score_threshold = 0.5
    num_subsamples_per_image = 10
    log_root = "logs"
    model_folder = "maskrcnn_2020-11-10_17-31-25"

    device = "cuda"

    model_folder_path = Path(log_root) / model_folder
    result_folder_path = model_folder_path / "results"
    visualization_image_folder_path = result_folder_path / "images"

    result_folder_path.mkdir(exist_ok=True)
    visualization_image_folder_path.mkdir(exist_ok=True)

    parameter_file_path = result_folder_path / "parameters.yaml"

    with open(parameter_file_path, "w") as file:
        yaml.dump(
            {
                "score_threshold": score_threshold,
                "num_subsamples_per_image": num_subsamples_per_image,
            },
            file,
        )

    best_model_path = get_best_model_path(model_folder_path, "model_")

    config_path = list(model_folder_path.glob("*.yml"))[0]

    config = Config.load(config_path)
    model = get_model(config["model"]["n_classes"])
    model.to(device)
    checkpoint = torch.load(best_model_path, map_location=device)
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
    model.eval()

    dataset_gt = Dataset(config["data"]["root_folder"], config["data"]["subset_validation"])

    masks_gt = []
    for _, target in dataset_gt:
        masks_gt_sample = list(target["masks"].cpu().numpy().astype("bool"))
        masks_gt_sample = filter_border_particles(masks_gt_sample)
        masks_gt += masks_gt_sample

    masks_pred = []
    scores_pred = []

    dataset_pred = Dataset(
        config["data"]["root_folder"],
        config["data"]["subset_validation"],
        transforms=get_transform(training=False),
    )

    dataloader = DataLoader(dataset_pred, batch_size=1)
    dataiterator = iter(dataloader)

    num_images = len(dataloader)

    num_subsamples = num_subsamples_per_image * num_images

    for subsample_id in range(num_subsamples):
        try:
            image, target = next(dataiterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            dataiterator = iter(dataloader)
            image, target = next(dataiterator)

        image = image.squeeze()

        with torch.no_grad():
            prediction = model([image.to(device)])[0]

        scores_pred_sample = list(prediction["scores"].cpu().numpy())
        masks_pred_sample = list(prediction["masks"].squeeze().round().cpu().numpy().astype("bool"))

        masks_pred_sample, scores_pred_sample = filter_border_particles(
            masks_pred_sample, scores_pred_sample
        )

        visualization_image_path = visualization_image_folder_path / f"{subsample_id}.png"

        save_visualization(
            image,
            {"masks": masks_pred_sample, "scores": scores_pred_sample},
            visualization_image_path,
            score_threshold=score_threshold,
            do_display_box=False,
            do_display_label=False,
            do_display_score=True,
        )

        masks_pred += masks_pred_sample
        scores_pred += scores_pred_sample

    area_equivalent_diameters_pred = calculate_area_equivalent_diameters(masks_pred)
    area_equivalent_diameters_gt = calculate_area_equivalent_diameters(masks_gt)

    compare_particle_size_distributions(
        area_equivalent_diameters_gt,
        area_equivalent_diameters_pred,
        scores_pred,
        measurand_name="Area Equivalent Diameter",
    )

    particle_size_distribution_path = result_folder_path / "particlesizedistribution.pdf"
    plt.savefig(particle_size_distribution_path)


def calculate_area_equivalent_diameters(masks):
    masks = np.array(masks)
    masks = masks.reshape(masks.shape[0], -1)
    areas = np.asarray(masks).sum(axis=1)
    return list(np.sqrt(4 * areas / np.pi))


def compare_particle_size_distributions(
    ground_truth, prediction, scores=None, measurand_name="Diameter", unit="px"
):
    colors = get_viridis_colors(2)

    hist_kwargs = {"density": True, "histtype": "step"}

    geometric_mean_gt = gmean(ground_truth)
    geometric_mean_pred = gmean(prediction, weights=scores)

    geometric_std_gt = gstd(ground_truth)
    geometric_std_pred = gstd(prediction, weights=scores)

    label_gt = (
        f"Ground Truth\n  $d_g={geometric_mean_gt:.0f}$ {unit}\n  $\sigma_g={geometric_std_gt:.2f}$"
    )
    label_pred = f"Prediction\n  $d_g={geometric_mean_pred:.0f}$ {unit}\n  $\sigma_g={geometric_std_pred:.2f}$"

    p_gt, bins, _ = plt.hist(ground_truth, color=colors[0], label=label_gt, **hist_kwargs)
    q_pred, _, _ = plt.hist(
        prediction, bins=bins, color=colors[1], weights=scores, label=label_pred, **hist_kwargs
    )
    plt.ylabel("")
    plt.legend()

    plt.xlabel(f"{measurand_name}/{unit}")

    plt.ylabel("Probability Density")


if __name__ == "__main__":
    inspect_model()

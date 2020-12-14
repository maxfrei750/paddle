from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

import fire
from data import Dataset
from deployment import analyze_image, load_trained_model
from postprocessing import calculate_area_equivalent_diameters, filter_border_particles
from torch.utils.data import DataLoader
from transforms import get_transform
from utilities import AnyPath, get_last_checkpoint_path, log_parameters_as_yaml, set_random_seed
from visualization import plot_particle_size_distributions, save_visualization


def inspect_model(
    model_id: str,
    log_root: AnyPath = "logs",
    data_root: AnyPath = "data",
    subset: str = "validation",
    cropping_rectangle: List[int] = None,
    num_subsamples_per_image: int = 1,
    score_threshold: float = 0.5,
    random_seed: int = 42,
    device: str = "cuda",
):
    """Inspect a model by applying it to a validation dataset and store the results in the model folder.

    :param model_id: Identifier of the model.
    :param log_root: Path of the log folder. Default: "logs"
    :param data_root: Path of the data folder. Default: "data"
    :param subset: Name of the subset to use for the validation. Default: "validation"
    :param cropping_rectangle: If not None, rectangle to use for the cropping of the validation images. Default: None
    :param num_subsamples_per_image: Number of images that are sampled from each validation sample image. Default: 1
    :param score_threshold: Detections with a score<score_threshold are discarded. Default: 0.5
    :param random_seed: Random seed for random, numpy and torch. Default: 42
    :param device: Computation device. Default: "cuda"
    """

    set_random_seed(random_seed)

    model_folder_path = Path(log_root) / model_id
    result_folder_path = model_folder_path / "results" / subset
    visualization_image_folder_path = result_folder_path / "images"

    result_folder_path.mkdir(exist_ok=True, parents=True)
    visualization_image_folder_path.mkdir(exist_ok=True)

    parameter_file_path = result_folder_path / "parameters.yaml"

    log_parameters_as_yaml(
        parameter_file_path,
        score_threshold=score_threshold,
        num_subsamples_per_image=num_subsamples_per_image,
        random_seed=random_seed,
    )

    model_path = get_last_checkpoint_path(model_folder_path)
    model = load_trained_model(model_path, device)

    dataset_gt = Dataset(data_root, subset)

    masks_gt = []
    for _, target in dataset_gt:
        masks_gt_sample = list(target["masks"].cpu().numpy().astype("bool"))
        masks_gt_sample = filter_border_particles({"masks": masks_gt_sample})["masks"]
        masks_gt += masks_gt_sample

    masks_pred = []
    scores_pred = []

    dataset_pred = Dataset(
        data_root,
        subset,
        transforms=get_transform(training=False, cropping_rectangle=cropping_rectangle),
    )

    dataloader_pred = DataLoader(dataset_pred, batch_size=1)
    dataiterator = iter(dataloader_pred)

    num_images = len(dataloader_pred)

    num_subsamples = num_subsamples_per_image * num_images

    for subsample_id in range(num_subsamples):
        try:
            image, target = next(dataiterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            dataiterator = iter(dataloader_pred)
            image, target = next(dataiterator)

        prediction = analyze_image(model, image)

        prediction = filter_border_particles(prediction)

        visualization_image_path = visualization_image_folder_path / f"{subsample_id}.png"

        save_visualization(
            image,
            prediction,
            visualization_image_path,
            score_threshold=score_threshold,
            do_display_box=False,
            do_display_label=False,
            do_display_score=True,
        )

        masks_pred += prediction["masks"]
        scores_pred += prediction["scores"]

    area_equivalent_diameters_pred = calculate_area_equivalent_diameters(masks_pred)
    area_equivalent_diameters_gt = calculate_area_equivalent_diameters(masks_gt)

    plot_particle_size_distributions(
        [area_equivalent_diameters_gt, area_equivalent_diameters_pred],
        score_lists=[None, scores_pred],
        labels=["Manual", "Prediction"],
        measurand_name="Area Equivalent Diameter",
    )

    particle_size_distribution_path = result_folder_path / "particlesizedistribution.pdf"
    plt.savefig(particle_size_distribution_path)


if __name__ == "__main__":
    fire.Fire(inspect_model)

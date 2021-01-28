from pathlib import Path

import pandas as pd
from transforms import get_transform

from data import Dataset
from deployment import analyze_image, load_trained_model
from postprocessing import (
    calculate_area_equivalent_diameters,
    calculate_maximum_feret_diameters,
    calculate_minimum_feret_diameters,
    filter_border_particles,
)
from utilities import set_random_seed
from visualization import visualize_detection

# TODO: Fix


def main():
    random_seed = 42

    device = "cuda"

    data_root = Path("data") / "sem"
    subset = "test"

    log_root = "logs"
    model_folder = "maskrcnn_2020-12-15_19-07-13"
    model_file_name = "checkpoint_MaskRCNN_1200.pt"

    set_random_seed(random_seed)

    model_folder_path = Path(log_root) / model_folder
    model_file_path = model_folder_path / model_file_name

    transform = get_transform(cropping_rectangle=[0, 0, 1280, 896])
    dataset = Dataset(data_root, subset, transforms=transform)

    model = load_trained_model(model_file_path, device)

    results = []

    for image, target in dataset:
        image_name = target["image_name"]

        prediction = analyze_image(model, image)
        prediction = filter_border_particles(prediction)

        # visualization = visualize_detection(
        #     image,
        #     prediction,
        #     do_display_box=False,
        #     do_display_label=False,
        #     do_display_score=False,
        #     score_threshold=0.5,
        # )
        #
        # visualization.show()

        area_equivalent_diameters = calculate_area_equivalent_diameters(prediction["masks"])
        feret_min_diameters = calculate_minimum_feret_diameters(prediction["masks"])
        feret_max_diameters = calculate_maximum_feret_diameters(prediction["masks"])

        result = pd.DataFrame(
            data={
                "image_name": image_name,
                "score": prediction["scores"],
                "area_equivalent_diameter": area_equivalent_diameters,
                "feret_min_diameter": feret_min_diameters,
                "feret_max_diameter": feret_max_diameters,
            }
        )

        results.append(result)

    results = pd.concat(results)

    results.to_csv(Path.cwd() / "results.csv")

    a = 1


if __name__ == "__main__":
    main()

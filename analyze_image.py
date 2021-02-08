from pathlib import Path

from PIL import Image

from data import MaskRCNNDataset
from deployment import analyze_image, load_trained_model
from postprocessing import filter_border_instances, filter_low_score_instances
from utilities import set_random_seed
from visualization import visualize_detection


def slice_image(image, slice_size=1024):

    _, image_height, image_width = image.shape

    assert (
        image_width % slice_size == 0 and image_height % slice_size == 0
    ), f"Image of size {image_height}x{image_width} cannot be sliced evenly with slice_size={slice_size}."

    steps_x = list(range(0, image_width + 1, slice_size))[:-1]
    steps_y = list(range(0, image_width + 1, slice_size))[:-1]

    image_slices = []

    for step_x in steps_x:
        for step_y in steps_y:
            image_slice = image[:, step_y : step_y + slice_size, step_x : step_x + slice_size]
            image_slices.append(image_slice)

    return image_slices


def main():
    score_threshold = 0.6
    random_seed = 42

    device = "cuda"

    data_root = "data"
    subset = "versatility_test_tem"

    log_root = "logs"
    model_folder = "maskrcnn_2020-11-10_17-31-25"
    model_file_name = "model_MaskRCNN_AP=0.0087.pt"

    set_random_seed(random_seed)

    do_invert_image = "sem" in subset
    do_slice_image = "tem" in subset

    model_folder_path = Path(log_root) / model_folder
    model_file_path = model_folder_path / model_file_name

    result_folder_path = model_folder_path / "results" / subset
    result_folder_path.mkdir(exist_ok=True)

    mask_folder_path = result_folder_path / "masks"
    mask_folder_path.mkdir(exist_ok=True)

    dataset = MaskRCNNDataset(data_root, subset)

    model = load_trained_model(model_file_path, device)

    for image, target in dataset:
        image_name = target["image_name"]

        if do_invert_image:
            image = 1 - image

        if do_slice_image:
            image_slices = slice_image(image)
        else:
            image_slices = [image]

        for image_id, image_slice in enumerate(image_slices):

            prediction = analyze_image(model, image_slice)
            prediction = filter_low_score_instances(prediction, score_threshold)
            prediction = filter_border_instances(prediction)

            visualization_image_path = (
                result_folder_path / f"visualization_{image_name}_{image_id}.png"
            )

            visualization = visualize_detection(
                image_slice,
                prediction,
                do_display_box=False,
                do_display_label=False,
                do_display_score=False,
            )

            visualization.save(
                visualization_image_path,
            )

            for mask_id, mask in enumerate(prediction["masks"]):
                mask_path = mask_folder_path / f"mask_{image_name}_{image_id}_{mask_id}.png"
                Image.fromarray(mask).save(mask_path)


if __name__ == "__main__":
    main()

import math
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import integrate, interpolate
from scipy.integrate import AccuracyWarning

# TODO: Type hints
# TODO: doc strings


def _remove_duplicate_keypoints(keypoints, weights):
    data = pd.DataFrame({"x": keypoints[:, 0], "y": keypoints[:, 1], "w": weights})

    is_duplicate = data.duplicated(subset=["x", "y"], keep=False)
    duplicates = data[is_duplicate]
    data = data.drop_duplicates(subset=["x", "y"])
    duplicates = duplicates.groupby(["x", "y"]).mean().reset_index()

    for _, duplicate in duplicates.iterrows():
        data.loc[(data["x"] == duplicate["x"]) & (data["y"] == duplicate["y"]), ["w"]] = duplicate[
            "w"
        ]

    keypoints = data[["x", "y"]].to_numpy()
    weights = data["w"].to_numpy()

    return keypoints, weights


def _filter_keypoints_based_on_score(keypoints, weights, threshold=0.5):
    data = pd.DataFrame({"x": keypoints[:, 0], "y": keypoints[:, 1], "w": weights})
    data = data[data["w"] >= threshold]

    keypoints = data[["x", "y"]].to_numpy()
    weights = data["w"].to_numpy()

    return keypoints, weights


def interpolation(keypoints, num_interpolation_steps, weights=None):
    tck = _prepare_interpolation(keypoints, weights)

    x_new, y_new = interpolate.splev(np.linspace(0, 1, num_interpolation_steps), tck, der=0)

    return np.stack((x_new, y_new), axis=1)


def _prepare_interpolation(keypoints, weights=None):
    if weights is None:
        weights = np.ones(len(keypoints))
    keypoints, weights = _remove_duplicate_keypoints(keypoints, weights)
    # keypoints, weights = _filter_keypoints_based_on_score(keypoints, weights)  # didn't work well
    # perform nearest neighbor plausibility check
    # https://stackoverflow.com/a/32781737/11652760
    if len(keypoints) < 4:
        spline_degree = 1
    else:
        spline_degree = 3
    tck, _ = interpolate.splprep(keypoints.T, s=0, k=spline_degree, w=weights)
    return tck


def to_mask(image_size, keypoints, width, num_interpolation_steps=100, weights=None):
    width = int(np.round(width))
    mask = Image.new("F", image_size[::-1])

    if num_interpolation_steps is not None:
        keypoints = interpolation(keypoints, num_interpolation_steps, weights=weights)
        keypoints = keypoints.astype(np.float32)

    keypoints = [tuple(x) for x in keypoints]
    ImageDraw.Draw(mask).line(keypoints, fill=1, width=width)

    # Draw ellipses at the line joins to cover up gaps.
    r = math.floor(width / 2) - 1

    for keypoint in keypoints[1:-1]:
        x, y = keypoint
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r

        ImageDraw.Draw(mask).ellipse([x0, y0, x1, y1], fill=1)

    return np.array(mask)


def calculate_length(keypoints, weights=None):
    tck = _prepare_interpolation(keypoints, weights)

    def length_function(u):
        x_der, y_der = interpolate.splev(u, tck, der=1)
        return np.sqrt(x_der ** 2 + y_der ** 2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AccuracyWarning)
        length = integrate.romberg(length_function, 0, 1)

    return length

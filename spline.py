from PIL import Image, ImageDraw
import math
import numpy as np
from scipy import interpolate


class Spline:
    def __init__(self, points=None, width=1):
        self._width = None
        self.width = width
        self.points_raw = points

    def get_mask(self, image_size):
        mask = Image.new("L", image_size, 0)

        if self.points_raw is not None:
            points = self.points_interpolated

            points = [tuple(x) for x in points]
            ImageDraw.Draw(mask).line(points, fill=255, width=self.width)

            # Draw ellipses at the line joins to cover up gaps.
            r = math.floor(self.width / 2) - 1.5

            for point in points[1:-1]:
                x, y = point
                x0 = x - r
                y0 = y - r
                x1 = x + r
                y1 = y + r

                ImageDraw.Draw(mask).ellipse([x0, y0, x1, y1], fill=255)

        return np.array(mask)

    @property
    def points_interpolated(self):
        return _spline_interpolation(self.points_raw)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        if value < 1:
            self._width = 1
        else:
            self._width = value


def _spline_interpolation(coordinates):
    n_interpolation_steps = 8
    n_coordinates = len(coordinates)

    if n_coordinates >= 4:
        tck, _ = interpolate.splprep(coordinates.T, s=0)
        x_new, y_new = interpolate.splev(np.linspace(0, 1, n_coordinates * n_interpolation_steps), tck, der=0)

        return np.stack((x_new, y_new), axis=-1).tolist()
    else:
        return coordinates

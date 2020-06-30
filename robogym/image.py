#  Copyright 2016-2020 Ivan Alles. See also the LICENSE file.

import numpy as np

import robogym.geometry as geo


def hsv_hue_diff(hue1, hue2, min_hue=1, max_hue=None):
    """
    Compute the difference between 2 hue values in HSV color space.

    :param hue1: first hue
    :param hue2: second hue
    :param min_hue: minimal hue value unless max_hue is None, in which case this is the max_hue, and min_hue is 0.
    :param max_hue: maximal hue value.
    :return: a value in [min_hue, max_hue] corresponding to the shortest distance from hue1 to hue2.
    """
    if max_hue is None:
        max_hue = min_hue
        min_hue = 0
    d = hue1 - hue2
    d = np.stack([d, -d])
    d = geo.normalize_angle(d, max_hue + min_hue, min_hue)
    d = np.min(d, axis=0)
    return d


def normalize_intensity(image, min, max, k, g, c, b, out=None):
    """
    Clip the intensity of the image at min, max. Then transform the values using the formula:
     ((k*image)^g)*c + b.

    Any of the normalization parameters can be None to skip the corresponding transformation.

    :param image: a numpy array.
    :param k a normalization constant to transform the values into the range [0..1].
    :param g gamma correction (0..+inf).
    :param c contrast.
    :param b brightness.
    :param out optional output tensor for in-place operation.
    :return normalized image.
    """
    changed = False

    # Clip first, this simulates natural image range (e.g. [0..255] or [0..1])
    # after data augmentation.
    if min is not None or max is not None:
        image, changed = np.clip(image, min, max, out=out), True

    # Transform ((k*image)^g)*c + b
    if k is not None:
        image, changed = np.multiply(image, k, out=out), True
    if g is not None:
        image, changed = np.power(image, g, out=out), True
    if c is not None:
        image, changed = np.multiply(image, c, out=out), True
    if b is not None:
        image, changed = np.add(image, b, out=out), True

    # In case all transform parameters are None
    if not changed and out is not None:
        out[:] = image
        image = out

    return image

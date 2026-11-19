"""
Fusion: combine several aligned (rectified) images into one.
"""

import warnings

import numpy as np


def fuse_mean(images):
    # fusion using mean
    stack = np.stack(images, axis=0)          # N x H x W x 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(stack, axis=0)


def fuse_median(images):
    # fusion using median
    stack = np.stack(images, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def to_uint8(image):
    # converts a float image (possibly with NaN) to an 8-bit image
    # NaN (no data) becomes black
    image = np.nan_to_num(image, nan=0.0)
    return np.clip(np.round(image), 0, 255).astype(np.uint8)
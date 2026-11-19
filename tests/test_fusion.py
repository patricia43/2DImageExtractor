import numpy as np

from mfr.fusion import fuse_mean, fuse_median, to_uint8


def test_mean_and_median():
    a = np.full((2, 2, 3), 10.0)
    b = np.full((2, 2, 3), 20.0)
    c = np.full((2, 2, 3), 90.0)      # an outlier

    assert np.allclose(fuse_mean([a, b, c]), 40.0)     # (10 + 20 + 90) / 3
    assert np.allclose(fuse_median([a, b, c]), 20.0)   # the middle value


def test_nan_is_ignored():
    a = np.full((2, 2, 3), 10.0)
    b = np.full((2, 2, 3), np.nan)    # no data in this image

    assert np.allclose(fuse_mean([a, b]), 10.0)
    assert np.allclose(fuse_median([a, b]), 10.0)


def test_to_uint8():
    img = np.array([[[-5.0, 100.4, 300.0]]])
    img[0, 0, 0] = np.nan
    out = to_uint8(img)
    assert out.dtype == np.uint8
    assert out.tolist() == [[[0, 100, 255]]]
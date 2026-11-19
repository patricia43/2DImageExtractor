import cv2
import numpy as np

from mfr.fusion import to_uint8
from mfr.rectify import rectify, rectify_homography, warp_to_canvas


def test_rectify_full_frame_gives_same_image():
    # If the "poster" corners are exactly the frame corners,
    # rectifying to the same size must give back the same image.
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(60, 80, 3), dtype=np.uint8)
    corners = [[0, 0], [79, 0], [79, 59], [0, 59]]

    out = rectify(frame, corners, 80, 60)

    assert out.shape == (60, 80, 3)
    assert not np.isnan(out).any()
    assert np.allclose(out, frame, atol=1)


def test_rectify_marks_missing_pixels_as_nan():
    # Corners that go outside the frame: part of the output has no data.
    frame = np.full((60, 80, 3), 100, dtype=np.uint8)
    corners = [[-40, 0], [79, 0], [79, 59], [-40, 59]]

    out = rectify(frame, corners, 120, 60)

    assert np.isnan(out[:, :30]).all()        # left part: outside the frame
    assert not np.isnan(out[:, 50:]).any()    # right part: inside


def test_composed_warp_equals_two_warps():
    # Warping with M and then with S must land the pixels in the same place
    # as warping once with the product S @ M.
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, size=(100, 120, 3), dtype=np.uint8)
    frame = cv2.GaussianBlur(frame, (9, 9), 0)        # smooth, so interpolation is gentle
    M = rectify_homography([[10, 5], [110, 12], [105, 95], [8, 90]], 100, 80)
    S = np.array([[1, 0, 1.5], [0, 1, -1.0], [0, 0, 1]])   # small extra shift

    two_steps = warp_to_canvas(to_uint8(warp_to_canvas(frame, M, 100, 80)), S, 100, 80)
    one_step = warp_to_canvas(frame, S @ M, 100, 80)

    inner = (slice(10, -10), slice(10, -10))
    assert np.abs(two_steps[inner] - one_step[inner]).mean() < 2
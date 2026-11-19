import cv2
import numpy as np
from skimage import data

from mfr.align import ecc_align, ecc_find_warp


def test_ecc_undoes_a_small_shift():
    # Take a real image, shift it by a known amount (3.5 px right, 2 px down),
    # and check that ECC moves it back onto the original.
    reference = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    shift = np.float32([[1, 0, 3.5], [0, 1, 2.0]])
    shifted = cv2.warpAffine(reference, shift, (512, 512))

    aligned, cc = ecc_align(reference, shifted)

    assert cc is not None and cc > 0.95
    # compare the central part (the borders have no data after shifting)
    ref_c = reference[50:-50, 50:-50].astype(np.float32)
    before = np.abs(shifted[50:-50, 50:-50].astype(np.float32) - ref_c).mean()
    after = np.abs(aligned[50:-50, 50:-50] - ref_c).mean()
    assert after < before / 3          # much closer to the reference than before


def test_ecc_find_warp_measures_the_shift():
    # The shifted image is the reference moved 3.5 px right and 2 px down,
    # so W (reference -> image coordinates) should contain that translation.
    reference = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    shift = np.float32([[1, 0, 3.5], [0, 1, 2.0]])
    shifted = cv2.warpAffine(reference, shift, (512, 512))

    W, cc = ecc_find_warp(reference, shifted)

    assert W is not None
    assert abs(W[0, 2] - 3.5) < 0.2       # x shift
    assert abs(W[1, 2] - 2.0) < 0.2       # y shift
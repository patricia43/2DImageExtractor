"""
Alignment: fine-tune an image so it overlaps a reference image exactly,
using ECC (Enhanced Correlation Coefficient).

Used for two things:
  1. aligning every rectified frame to the reference frame before fusion
  2. registering a final result to the original poster before computing
     PSNR/SSIM (so we measure image quality, not click position)
"""

import cv2
import numpy as np


def ecc_find_warp(reference, image, margin=0.1, iterations=200):
    """Find the correction between `image` and `reference` with ECC,
    WITHOUT warping anything.

    reference, image : same size, H x W x 3, float32 or uint8 (NaN allowed).
    margin           : ECC only looks at the central part of the images
                       (here: without 10% on each side), where both images
                       surely have data.
    returns          : (W, cc)
                       W  = 3x3 matrix that maps reference coordinates ->
                            image coordinates
                       cc = correlation after alignment (1.0 = perfect match)
                       If ECC fails: (None, None)
    """
    reference = np.nan_to_num(reference.astype(np.float32), nan=0.0)
    image = np.nan_to_num(image.astype(np.float32), nan=0.0)
    h, w = reference.shape[:2]

    # ECC works on grayscale images, and only on the central crop
    y0, y1 = int(h * margin), int(h * (1 - margin))
    x0, x1 = int(w * margin), int(w * (1 - margin))
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]

    # Start from "no correction" (identity) and let ECC improve it
    warp = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, 1e-6)
    try:
        cc, warp = cv2.findTransformECC(ref_gray, img_gray, warp,
                                        cv2.MOTION_HOMOGRAPHY, criteria, None, 5)
    except cv2.error:
        return None, None      # ECC did not converge (images too different)

    # ECC worked on the crop; convert the correction to full-image coordinates:
    # shift to crop coords -> apply correction -> shift back
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], dtype=np.float64)
    W = T @ warp.astype(np.float64) @ np.linalg.inv(T)
    return W, float(cc)


def ecc_align(reference, image, margin=0.1, iterations=200):
    """Warp `image` so that it matches `reference`.

    returns : (aligned image as float32 with NaN where there is no data, cc).
              If ECC fails, the image is returned unchanged and cc is None.
    """
    W, cc = ecc_find_warp(reference, image, margin, iterations)
    if W is None:
        return image.astype(np.float32).copy(), None

    image_f = image.astype(np.float32)
    valid = ~np.isnan(image_f).any(axis=2)          # True where image has data
    image_f = np.nan_to_num(image_f, nan=0.0)
    h, w = reference.shape[:2]

    # W maps reference coordinates -> image coordinates,
    # so we use WARP_INVERSE_MAP to pull the image onto the reference grid
    flags = cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
    aligned = cv2.warpPerspective(image_f, W, (w, h), flags=flags)

    # Move the "has data" mask the same way, and put NaN back where needed
    valid_u8 = valid.astype(np.uint8) * 255
    valid_aligned = cv2.warpPerspective(valid_u8, W, (w, h), flags=flags)
    aligned[valid_aligned < 255] = np.nan

    return aligned, cc
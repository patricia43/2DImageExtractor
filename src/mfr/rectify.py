"""
Rectification: turn the skewed poster in a video frame into a frontal image.
"""

import cv2
import numpy as np


def canvas_corners(width, height):
    """Corners of the output (frontal) image, in the same order we always use:
    top-left, top-right, bottom-right, bottom-left."""
    return np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])


def rectify_homography(corners, width, height):
    """The 3x3 homography that takes the 4 poster corners in the frame
    to the 4 corners of the frontal canvas (frame -> canvas)."""
    src = np.float32(corners)
    dst = canvas_corners(width, height)
    return cv2.getPerspectiveTransform(src, dst)


def warp_to_canvas(frame, M, width, height):
    """Warp `frame` with the 3x3 matrix M (frame -> canvas).

    returns : float32 image (height x width x 3). Pixels for which the
              frame has no data (outside the frame) are set to NaN, so the
              fusion step can ignore them.
    """
    warped = cv2.warpPerspective(frame, M, (width, height), flags=cv2.INTER_LINEAR)
    warped = warped.astype(np.float32)

    # Which canvas pixels actually came from inside the frame?
    # Warp an all-white image: 255 = fully inside, less than 255 = outside or
    # on the border (partly mixed with the black outside), so we drop those.
    ones = np.full(frame.shape[:2], 255, dtype=np.uint8)
    valid = cv2.warpPerspective(ones, M, (width, height), flags=cv2.INTER_LINEAR)
    warped[valid < 255] = np.nan

    return warped


def rectify(frame, corners, width, height):
    """Warp the poster from `frame` into a frontal image of size width x height.

    frame   : the video frame (H x W x 3, uint8)
    corners : the 4 poster corners in the frame, [[x, y], ...]
              order: top-left, top-right, bottom-right, bottom-left
    returns : float32 image with NaN where there is no data
    """
    M = rectify_homography(corners, width, height)
    return warp_to_canvas(frame, M, width, height)
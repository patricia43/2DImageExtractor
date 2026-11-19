"""
Minimal synthetic video generator.

Takes a known image, moves it in front of a virtual camera
along a smooth path, adds noise, and saves:
  - the video as an MP4
  - every frame as a PNG (lossless)
  - the ground truth: the original image + the poster corners and the
    homography for every frame
"""

import json
import os

import cv2
import numpy as np
from skimage import data

# Settings

OUT_DIR = "data/synthetic/astronaut_sigma20"

N_FRAMES = 90
FPS = 30
VIDEO_W, VIDEO_H = 1280, 720

NOISE_SIGMA = 20.0         # strength of the Gaussian noise (0 = no noise)
SHAKE_PX = 1.5            # how much the hand shakes, in pixels

SEED = 0


# Where the 4 poster corners are in the video at a few "key" moments.
# Order of corners: top-left, top-right, bottom-right, bottom-left.
KEYFRAMES = [
    [[330, 110], [760, 170], [760, 560], [330, 620]],
    [[420, 130], [860, 140], [855, 590], [425, 600]],
    [[520, 170], [940, 110], [940, 630], [520, 570]],
    [[440, 70], [960, 100], [940, 660], [420, 630]],
]


def load_poster():
    img = data.astronaut()                        # 512 x 512 x 3, RGB
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def make_background(rng):
    # randomly generate a blurry texture for the background to simulate real wall
    noise = rng.integers(0, 256, size=(VIDEO_H // 8, VIDEO_W // 8, 3), dtype=np.uint8)
    bg = cv2.resize(noise, (VIDEO_W, VIDEO_H), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (31, 31), 0)
    # make it darker and less colorful so the poster stands out
    bg = (0.4 * bg + 60).astype(np.uint8)
    return bg


def interpolate_corners(keyframes, n_frames):
    # return an array of shape (n_frames, 4, 2): the corners in every frame

    # for each frame we find between which two keyframes we are and blend
    # them with a cosine curve to simulate real camera movement
    keyframes = np.array(keyframes, dtype=np.float32)
    n_keys = len(keyframes)
    corners = np.zeros((n_frames, 4, 2), dtype=np.float32)

    for t in range(n_frames):
        # position along the keyframes, from 0 to n_keys - 1
        s = t / (n_frames - 1) * (n_keys - 1)
        i = min(int(s), n_keys - 2) # index of the keyframe before us
        u = s - i                   # 0 = at keyframe i, 1 = at keyframe i+1
        u_smooth = (1 - np.cos(np.pi * u)) / 2
        corners[t] = (1 - u_smooth) * keyframes[i] + u_smooth * keyframes[i + 1]

    return corners


def make_hand_shake(rng, n_frames):
    # small random offsets that change slowly from frame to frame
    # returns an array of shape (n_frames, 2) with (dx, dy) per frame
    raw = rng.normal(0, 1, size=(n_frames, 2))
    kernel = np.ones(7) / 7
    smooth = np.zeros_like(raw)
    for k in range(2):
        smooth[:, k] = np.convolve(raw[:, k], kernel, mode="same")
    # rescale so the typical shake is about SHAKE_PX pixels
    smooth = smooth / (smooth.std() + 1e-8) * SHAKE_PX
    return smooth


def render_frame(poster, background, H):
    # Warp the poster with homography H and paste it onto the background
    warped = cv2.warpPerspective(poster, H, (VIDEO_W, VIDEO_H))

    # warp an all-white image the same way: white = pixels covered by the poster
    ones = np.full(poster.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(ones, H, (VIDEO_W, VIDEO_H))

    # Blend with the mask as a weight (0..1)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    frame = alpha * warped + (1 - alpha) * background
    return frame


def add_noise(frame, rng):
    # add Gaussian noise and convert back to a normal 8-bit image
    noisy = frame + rng.normal(0, NOISE_SIGMA, size=frame.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(os.path.join(OUT_DIR, "frames"), exist_ok=True)

    poster = load_poster()
    background = make_background(rng)
    h, w = poster.shape[:2]

    # Corners of the original poster image (same order as KEYFRAMES)
    src_corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    # Where the corners are in each video frame = smooth path + hand shake
    corners = interpolate_corners(KEYFRAMES, N_FRAMES)
    shake = make_hand_shake(rng, N_FRAMES)
    corners = corners + shake[:, None, :]          # same shake for all 4 corners

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(OUT_DIR, "video.mp4")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (VIDEO_W, VIDEO_H))

    gt_frames = []
    for t in range(N_FRAMES):
        # Homography: poster image coordinates -> video frame coordinates
        H = cv2.getPerspectiveTransform(src_corners, corners[t].astype(np.float32))

        frame = render_frame(poster, background, H)
        frame = add_noise(frame, rng)

        cv2.imwrite(os.path.join(OUT_DIR, "frames", f"frame_{t:03d}.png"), frame)
        writer.write(frame)

        gt_frames.append({
            "index": t,
            "corners": corners[t].tolist(),        # 4 x [x, y] in the video frame
            "H": H.tolist(),                       # 3 x 3, poster -> frame
        })

    writer.release()

    # Ground truth: the clean original + everything needed to check results
    cv2.imwrite(os.path.join(OUT_DIR, "original.png"), poster)
    ground_truth = {
        "poster_size": [w, h],
        "video_size": [VIDEO_W, VIDEO_H],
        "fps": FPS,
        "noise_sigma": NOISE_SIGMA,
        "shake_px": SHAKE_PX,
        "seed": SEED,
        "frames": gt_frames,
    }
    with open(os.path.join(OUT_DIR, "ground_truth.json"), "w") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Saved {N_FRAMES} frames, video and ground truth to {OUT_DIR}")


if __name__ == "__main__":
    main()
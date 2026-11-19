"""
Experiment on the synthetic video:
single frame vs. mean/median fusion, with and without ECC alignment,
for different click errors.

For each chosen frame we take the TRUE poster corners from ground_truth.json,
add a random error (to imitate a person clicking) and rectify the frame.
Then:
  - "no ECC":   fuse the rectified frames directly
  - "ECC":      measure with ECC how every frame differs from the first
                one (the reference), combine that correction with the
                rectification into ONE matrix, and warp each original
                video frame only once. Then fuse.

Evaluation: every result is first registered to the original poster with
ECC (so we measure image quality, not where the clicks put the poster),
then compared with PSNR and SSIM (higher = closer to the original).

Run from the project root (after make_synthetic.py):
    python scripts/run_synthetic.py
"""

import csv
import json
import os

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from mfr.align import ecc_align, ecc_find_warp
from mfr.fusion import fuse_mean, fuse_median, to_uint8
from mfr.rectify import rectify_homography, warp_to_canvas


# Settings

DATA_DIR = "data/synthetic/astronaut"
OUT_DIR = "results/synthetic"

FRAMES = [0, 20, 45, 70, 89]          # which frames to use; the first is the reference
CLICK_ERRORS_PX = [0, 1, 2, 4]        # simulated click error (std, in pixels)
SEED = 0

EVAL_BORDER = 20                      # ignore this many pixels at each edge when measuring
ZOOM = (150, 40, 160, 160)            # zoomed region for the pictures: the astronaut's face


def simulate_clicks(true_corners, error_px, rng):
    # true corners + random error
    true_corners = np.array(true_corners, dtype=np.float32)
    error = rng.normal(0, error_px, size=true_corners.shape)
    return true_corners + error


def evaluate(original, result):
    # register result to the original, then compute PSNR and SSIM.
    # returns (psnr, ssim, registered image as uint8)
    registered, _ = ecc_align(original, result)
    registered = to_uint8(registered)

    # the edges may have no data after registering, so we skip them
    b = EVAL_BORDER
    orig_c = original[b:-b, b:-b]
    res_c = registered[b:-b, b:-b]

    psnr = peak_signal_noise_ratio(orig_c, res_c, data_range=255)
    ssim = structural_similarity(orig_c, res_c, channel_axis=2, data_range=255)
    return round(float(psnr), 2), round(float(ssim), 3), registered


def zoom(image):
    x, y, w, h = ZOOM
    crop = image[y:y + h, x:x + w]
    return cv2.resize(crop, (2 * w, 2 * h), interpolation=cv2.INTER_NEAREST)


def labeled(image, text):
    """Copy of the image with a label written on it."""
    image = image.copy()
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(os.path.join(DATA_DIR, "ground_truth.json")) as f:
        gt = json.load(f)
    original = cv2.imread(os.path.join(DATA_DIR, "original.png"))
    width, height = gt["poster_size"]

    # load the chosen frames (lossless PNGs)
    frames = {}
    for t in FRAMES:
        path = os.path.join(DATA_DIR, "frames", f"frame_{t:03d}.png")
        frames[t] = cv2.imread(path)

    rows = []
    for error_px in CLICK_ERRORS_PX:
        rng = np.random.default_rng(SEED)
        print(f"\n--- click error {error_px} px ---")

        # 1) rectify every chosen frame using (noisy) clicked corners.
        #    We keep the matrices M, we'll need them again in step 2.
        matrices = []
        rectified = []
        for t in FRAMES:
            clicks = simulate_clicks(gt["frames"][t]["corners"], error_px, rng)
            M = rectify_homography(clicks, width, height)        # frame -> canvas
            matrices.append(M)
            rectified.append(warp_to_canvas(frames[t], M, width, height))

        # 2) ECC: measure how each frame differs from the reference (the first
        #    frame), then warp the ORIGINAL video frame once with the combined
        #    matrix inv(W) @ M  (first M: frame -> its canvas,
        #                        then inv(W): its canvas -> reference canvas)
        reference = rectified[0]
        aligned = [reference]
        for i in range(1, len(FRAMES)):
            t = FRAMES[i]
            W, cc = ecc_find_warp(reference, rectified[i])
            if W is None:
                print(f"  frame {t}: ECC failed, using it without alignment")
                aligned.append(rectified[i])
                continue
            print(f"  frame {t}: ECC cc = {cc:.4f}")
            combined = np.linalg.inv(W) @ matrices[i]
            aligned.append(warp_to_canvas(frames[t], combined, width, height))

        # 3) single-frame baseline: the BEST individual frame
        #    (fair comparison: we don't pick a bad frame on purpose)
        single_results = [evaluate(original, to_uint8(r)) for r in rectified]
        best = int(np.argmax([res[0] for res in single_results]))
        single = single_results[best]

        # fusion, without and with ECC
        mean_no_ecc = evaluate(original, to_uint8(fuse_mean(rectified)))
        median_no_ecc = evaluate(original, to_uint8(fuse_median(rectified)))
        mean_ecc = evaluate(original, to_uint8(fuse_mean(aligned)))
        median_ecc = evaluate(original, to_uint8(fuse_median(aligned)))

        row = {
            "click_error_px": error_px,
            "n_frames": len(FRAMES),
            "psnr_single_best": single[0],
            "psnr_mean_no_ecc": mean_no_ecc[0],
            "psnr_median_no_ecc": median_no_ecc[0],
            "psnr_mean_ecc": mean_ecc[0],
            "psnr_median_ecc": median_ecc[0],
            "ssim_single_best": single[1],
            "ssim_mean_no_ecc": mean_no_ecc[1],
            "ssim_median_no_ecc": median_no_ecc[1],
            "ssim_mean_ecc": mean_ecc[1],
            "ssim_median_ecc": median_ecc[1],
        }
        rows.append(row)
        print(f"  PSNR  single {single[0]} | mean no-ECC {mean_no_ecc[0]} | "
              f"mean ECC {mean_ecc[0]} | median ECC {median_ecc[0]}")
        print(f"  SSIM  single {single[1]} | mean no-ECC {mean_no_ecc[1]} | "
              f"mean ECC {mean_ecc[1]} | median ECC {median_ecc[1]}")

        # save comparison picture: full on top, zoomed face below
        pictures = [
            (original, "original"),
            (single[2], f"single (frame {FRAMES[best]})"),
            (mean_no_ecc[2], "mean, no ECC"),
            (mean_ecc[2], "mean, ECC"),
            (median_ecc[2], "median, ECC"),
        ]
        top = np.hstack([labeled(img, text) for img, text in pictures])
        bottom = np.hstack([zoom(img) for img, _ in pictures])
        bottom = cv2.resize(bottom, (top.shape[1], bottom.shape[0] * top.shape[1] // bottom.shape[1]))
        name = f"compare_error_{error_px}px.png"
        cv2.imwrite(os.path.join(OUT_DIR, name), np.vstack([top, bottom]))

    # save all numbers in a table
    with open(os.path.join(OUT_DIR, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
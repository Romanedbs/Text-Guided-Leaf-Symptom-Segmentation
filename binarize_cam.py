"""
Binarize normalized CAM matrices into coarse seed masks.

Corresponds to the binarization step described in Section 2.2.1 of the
manuscript.

NOTE: the manuscript states that pixels are binarized using an absolute
threshold on the [0, 1] normalized activation ("exceeds 0.99"). This
script's default ("percentile") instead keeps the top X% most activated
pixels *per image*, which is what the original notebook did (with
percentile_top=1, i.e. threshold = 99th percentile). These two methods
are NOT equivalent in general -- confirm which one actually produced the
reported results, and use --method absolute if you want to match the
manuscript's wording exactly.

Usage
-----
python binarize_cam.py \
    --cam_dir /path/to/cam_matrices \
    --output_dir /path/to/binary_masks \
    --method percentile --percentile_top 1
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd


def load_and_normalize_cam(csv_path):
    """Load a CAM matrix from .csv and rescale it to [0, 1]."""
    cam = (
        pd.read_csv(csv_path, header=None)
        .dropna(axis=1, how="all")
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
    )
    cam_min, cam_max = cam.min().min(), cam.max().max()
    return (cam - cam_min) / (cam_max - cam_min)


def binarize_percentile(cam, percentile_top=1):
    """Keep the top `percentile_top` % most activated pixels of this image."""
    threshold = np.percentile(cam, 100 - percentile_top)
    return (cam >= threshold).astype(np.uint8)


def binarize_absolute(cam, threshold=0.99):
    """Keep pixels whose normalized activation exceeds `threshold`."""
    return (cam >= threshold).astype(np.uint8)


def process_directory(cam_dir, output_dir, method="percentile",
                       percentile_top=1, threshold=0.99):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(cam_dir):
        cam_path = os.path.join(cam_dir, filename)
        cam = load_and_normalize_cam(cam_path)

        if method == "percentile":
            mask = binarize_percentile(cam, percentile_top)
        elif method == "absolute":
            mask = binarize_absolute(cam, threshold)
        else:
            raise ValueError(f"Unknown method: {method}")

        # CAM files are named "<original_image_name>.csv" -- stripping
        # ".csv" recovers the original image filename (with its extension),
        # so the mask can be matched to its image downstream.
        out_filename = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, out_filename)
        mask_to_save = (mask.to_numpy() * 255).astype(np.uint8)
        cv2.imwrite(out_path, mask_to_save)


def parse_args():
    parser = argparse.ArgumentParser(description="Binarize CAM matrices into seed masks")
    parser.add_argument("--cam_dir", required=True, help="Directory of CAM .csv matrices")
    parser.add_argument("--output_dir", required=True, help="Directory to save binary masks")
    parser.add_argument("--method", choices=["percentile", "absolute"], default="percentile")
    parser.add_argument("--percentile_top", type=float, default=1,
                         help="Keep top X%% most activated pixels (method=percentile)")
    parser.add_argument("--threshold", type=float, default=0.99,
                         help="Absolute activation threshold (method=absolute)")
    return parser.parse_args()


def main():
    args = parse_args()
    process_directory(
        args.cam_dir, args.output_dir,
        method=args.method,
        percentile_top=args.percentile_top,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()

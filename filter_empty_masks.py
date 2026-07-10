"""
Remove masks that are entirely empty (no predicted symptom pixels).

Used after text-guided / hybrid SAM3 pseudo-mask generation to build the
effective training set, since not all prompts produce a mask for every
image (see the manuscript: "images for which no prediction mask was
generated are excluded from training").

Usage
-----
python filter_empty_masks.py --input_dir ... --output_dir ...
"""

import argparse
import os
import shutil

import numpy as np
import skimage as ski


def filter_empty_masks(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    kept, discarded = 0, 0

    for root, _, files in os.walk(input_dir):
        for filename in files:
            path = os.path.join(root, filename)
            mask = ski.io.imread(path)

            if np.array_equal(np.unique(mask), np.array([0])):
                discarded += 1
                continue

            shutil.copy(path, os.path.join(output_dir, filename))
            kept += 1

    print(f"Kept: {kept}, discarded (empty): {discarded}")
    return kept, discarded


def parse_args():
    parser = argparse.ArgumentParser(description="Filter out empty (all-black) masks")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    filter_empty_masks(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

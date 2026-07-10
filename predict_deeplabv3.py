"""
Run inference with a trained DeepLabV3+ checkpoint and save predicted
masks for a test set.

Corresponds to the prediction step following training in Section 2.3 of
the manuscript. Segmentation metrics (IoU, Precision, Recall, rRMSE) are
computed separately -- see src/evaluation/metrics.py.

Usage
-----
python predict_deeplabv3.py \
    --checkpoint /path/to/model_best_loss.pth \
    --images_dir /path/to/test/images \
    --masks_dir /path/to/test/gt \
    --output_dir /path/to/save/predictions
"""

import argparse
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from train_deeplabv3 import (
    CLASS_RGB_VALUES, LeafSymptomDataset, build_model,
    colour_code_segmentation, get_preprocessing, reverse_one_hot,
)


def predict(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, preprocessing_fn = build_model(args.encoder, args.encoder_weights)
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval().to(device)

    dataset = LeafSymptomDataset(
        args.images_dir, args.masks_dir,
        preprocessing=get_preprocessing(preprocessing_fn, (args.image_size, args.image_size)),
        class_rgb_values=CLASS_RGB_VALUES,
        correction=args.exif_correction,
        return_filename=True,
    )
    loader = DataLoader(dataset, batch_size=1)

    for image, _gt_mask, filename in tqdm(loader, desc="Predicting"):
        x_tensor = image.to(device)
        with torch.no_grad():
            pred_mask = model(x_tensor)
        pred_mask = pred_mask.detach().cpu().numpy()[0]

        pred_mask = np.transpose(pred_mask, (1, 2, 0))
        pred_mask = colour_code_segmentation(reverse_one_hot(pred_mask), CLASS_RGB_VALUES)

        cv2.imwrite(os.path.join(args.output_dir, filename[0]), pred_mask)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict masks with a trained DeepLabV3+ model")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--masks_dir", required=True, help="Ground-truth masks (for the dataset loader; not used for metrics here)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--encoder", default="resnet101")
    parser.add_argument("--encoder_weights", default="imagenet")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--exif_correction", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    predict(parse_args())

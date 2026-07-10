"""
Refine CAM-derived seed masks into segmentation masks using SAM
(original Segment Anything Model), guided by box prompts.

Corresponds to the "CAM + SAM(box)" pipeline in Section 2.2.1 of the
manuscript. For each connected component of the binary seed mask, a
bounding box is derived and used as a box prompt for SAM.

Usage
-----
python sam_refine.py \
    --image_dir /path/to/images \
    --seed_mask_dir /path/to/binary_seed_masks \
    --output_dir /path/to/output_masks \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth
"""

import argparse
import os

import cv2
import numpy as np
import skimage as ski
import torch
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm


def seed_mask_to_boxes_input(seed_mask):
    """Convert a binary seed mask (255 = symptom, 0 = background) into a
    single-channel 0/255 image highlighting symptom regions, ready for
    contour detection."""
    mask = seed_mask.copy()
    mask[mask < 150] = 1   # background
    mask[mask >= 150] = 2  # symptom
    inverted = (1 - (mask == 1).astype(np.uint8)) * 255  # symptom -> 255
    if inverted.ndim == 3:
        inverted = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
    return inverted


def extract_boxes(inverted_mask, min_area=50):
    """Derive one bounding box per connected component of the symptom mask."""
    contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:
            boxes.append([x, y, x + w, y + h])
    return boxes


def refine_image(predictor, image, boxes, device):
    """Run SAM with box prompts and return the union of predicted masks,
    or None if no valid box is available."""
    if len(boxes) == 0:
        return None

    input_boxes = torch.tensor(boxes, dtype=torch.float32, device=device)
    predictor.set_image(image)
    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

    if transformed_boxes.numel() == 0:
        return None

    with torch.no_grad():
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

    global_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        binary_image = (mask.cpu().numpy().astype(np.uint8)[0]) * 255
        global_mask = cv2.bitwise_or(global_mask, binary_image)

    return global_mask


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["default"](checkpoint=args.sam_checkpoint).to(device)
    predictor = SamPredictor(sam)

    filenames = sorted(os.listdir(args.image_dir))

    for filename in tqdm(filenames, desc="SAM refinement"):
        try:
            image = ski.io.imread(os.path.join(args.image_dir, filename))
            seed_mask = ski.io.imread(os.path.join(args.seed_mask_dir, filename))

            image = cv2.resize(image, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
            seed_mask = cv2.resize(seed_mask, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

            inverted = seed_mask_to_boxes_input(seed_mask)
            boxes = extract_boxes(inverted, min_area=args.min_box_area)
            global_mask = refine_image(predictor, image, boxes, device)

            if global_mask is None:
                print(f"No valid box found for {filename}, skipping.")
                continue

            cv2.imwrite(os.path.join(args.output_dir, filename), global_mask)

        except Exception as e:
            print(f"Error on {filename}: {e}")
            continue

        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Refine seed masks with SAM (box prompt)")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--seed_mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sam_checkpoint", required=True, help="Path to sam_vit_h_4b8939.pth")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--min_box_area", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    main()

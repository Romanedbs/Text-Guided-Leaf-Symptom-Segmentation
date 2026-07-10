"""
Refine CAM-derived seed masks into segmentation masks using SAM3,
guided by box prompts, optionally combined with a text prompt.

Corresponds to the "CAM + SAM3(box)" and "CAM + SAM3(box+txt)" pipelines
described in the manuscript. Pass --text_prompt to run the hybrid
(box+txt) version; omit it to run the box-only version.

Usage
-----
# box only
python sam3_refine.py --image_dir ... --seed_mask_dir ... --output_dir ...

# hybrid box + text
python sam3_refine.py --image_dir ... --seed_mask_dir ... --output_dir ... \
    --text_prompt "leaf spot disease symptoms"
"""

import argparse
import gc
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import normalize_bbox


def seed_mask_to_boxes(seed_mask, min_area=1):
    """Derive bounding boxes (x, y, w, h) from a binary seed mask
    (255 = symptom, 0 = background)."""
    mask = seed_mask.copy()
    mask[mask < 150] = 1
    mask[mask >= 150] = 2
    inverted = (1 - (mask == 1).astype(np.uint8)) * 255
    if inverted.ndim == 3:
        inverted = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:
            boxes.append([x, y, w, h])
    return boxes


def refine_image(processor, image, boxes, text_prompt=None, device="cuda"):
    """Run SAM3 with box prompts (and optional text prompt), return the
    union mask (all zeros if no box and no mask is produced)."""
    width, height = image.size

    if len(boxes) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    input_boxes = torch.tensor(boxes, dtype=torch.float32, device=device)
    boxes_cxcywh = box_xywh_to_cxcywh(input_boxes.view(-1, 4))
    norm_boxes = normalize_bbox(boxes_cxcywh, width, height).tolist()
    box_labels = [True] * len(input_boxes)

    with torch.no_grad():
        state = processor.set_image(image)

        if text_prompt is not None:
            state = processor.set_text_prompt(state=state, prompt=text_prompt)

        for box, label in zip(norm_boxes, box_labels):
            gc.collect()
            torch.cuda.empty_cache()
            state = processor.add_geometric_prompt(state=state, box=box, label=label)

    masks = state.get("masks", [])
    if masks is None or len(masks) == 0:
        global_mask = np.zeros((height, width), dtype=np.uint8)
    else:
        global_mask = None
        for mask in masks:
            binary_image = (mask.cpu().numpy().astype(np.uint8)[0]) * 255
            if global_mask is None:
                global_mask = np.zeros_like(binary_image, dtype=np.uint8)
            global_mask = cv2.bitwise_or(global_mask, binary_image)

    processor.reset_all_prompts(state)
    return global_mask


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_sam3_image_model()
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    filenames = sorted(os.listdir(args.image_dir))

    for filename in tqdm(filenames, desc="SAM3 refinement"):
        image = Image.open(os.path.join(args.image_dir, filename))
        seed_mask = Image.open(os.path.join(args.seed_mask_dir, filename))
        seed_mask = np.array(seed_mask.resize(image.size, Image.NEAREST))

        boxes = seed_mask_to_boxes(seed_mask, min_area=args.min_box_area)
        global_mask = refine_image(
            processor, image, boxes,
            text_prompt=args.text_prompt,
            device=device,
        )

        cv2.imwrite(os.path.join(args.output_dir, filename), global_mask)
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Refine seed masks with SAM3 (box, optionally + text)")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--seed_mask_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_prompt", default=None,
                         help="If set, run the hybrid box+text pipeline with this prompt")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--min_box_area", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    main()

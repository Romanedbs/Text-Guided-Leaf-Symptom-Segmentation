"""
Generate pseudo-masks using SAM3 with text prompts only (no spatial
supervision).

Corresponds to the "SAM3(txt)" pipeline in the manuscript. Runs every
prompt in --prompts against every image in --image_dir and saves one
subfolder of masks per prompt.

NOTE: the manuscript reports 10 prompts, of which "leaf lesion" was
discarded for insufficient valid masks (9/10 retained for SAM3(txt)).
The original notebook also included an 11th prompt, "leaf symptom",
not mentioned in the manuscript -- it has been dropped from the
default list below; pass it explicitly via --prompts if you need it.

Usage
-----
python sam3_text_only.py \
    --image_dir /path/to/images \
    --output_dir /path/to/output \
    --prompts "lesion" "spot" "spots" "leaf spot" "leaf spots" \
              "leaf spot symptom" "leaf spot symptoms" \
              "leaf spot disease symptom" "leaf spot disease symptoms" \
              "leaf lesion"
"""

import argparse
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

DEFAULT_PROMPTS = [
    "lesion", "spot", "spots", "leaf spot", "leaf spots",
    "leaf spot symptom", "leaf spot symptoms",
    "leaf spot disease symptom", "leaf spot disease symptoms",
    "leaf lesion",
]


def run_prompt(model, image_dir, output_dir, prompt, confidence_threshold=0.5):
    prompt_output_dir = os.path.join(output_dir, prompt)
    os.makedirs(prompt_output_dir, exist_ok=True)

    filenames = sorted(os.listdir(image_dir))

    for filename in tqdm(filenames, desc=f"Prompt: {prompt}"):
        image = Image.open(os.path.join(image_dir, filename))

        processor = Sam3Processor(model, confidence_threshold=confidence_threshold)
        state = processor.set_image(image)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

        masks = state["masks"].cpu().numpy()  # (N, 1, H, W)
        masks = masks.squeeze(1)              # (N, H, W)

        global_mask = (masks > 0.5).any(axis=0).astype(np.uint8) * 255
        Image.fromarray(global_mask).save(os.path.join(prompt_output_dir, filename))


def main():
    args = parse_args()
    model = build_sam3_image_model()

    for prompt in args.prompts:
        run_prompt(model, args.image_dir, args.output_dir, prompt,
                   confidence_threshold=args.confidence_threshold)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text-only SAM3 pseudo-masks")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    main()

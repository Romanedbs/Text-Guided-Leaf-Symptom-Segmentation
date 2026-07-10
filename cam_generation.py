"""
Generate Class Activation Maps (CAMs) for leaf disease images using a
DinoV2-based classifier and ViT_CX.

Corresponds to Section 2.2.1 ("Image-level label-based pseudo-mask
generation") of the manuscript. Each CAM is saved as a normalized
(0-1) matrix in a .csv file, used downstream as the seed for SAM /
SAM3 refinement.

Usage
-----
python cam_generation.py \
    --input_dir /path/to/images \
    --output_dir /path/to/save/cams \
    --checkpoint /path/to/model_best.pth.tar \
    --class_map /path/to/class_map.txt
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset
from ViT_CX import ViT_CX
from tqdm import tqdm


class ImageDatasetWithFilename(Dataset):
    """Loads every image from a flat directory and returns (image, filename)."""

    VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(self.VALID_EXTENSIONS)
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        filename = os.path.basename(img_path)
        if self.transform:
            image = self.transform(image)
        return image, filename


def load_model(checkpoint_path, class_map_path, device):
    """Load the DinoV2/timm classifier and its class map from a checkpoint."""
    # Required for torch >= 2.6, where argparse.Namespace must be
    # explicitly allow-listed to unpickle checkpoints saved with
    # weights_only defaults.
    torch.serialization.add_safe_globals([argparse.Namespace])
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = timm.create_model(
        checkpoint["arch"], num_classes=checkpoint["args"].num_classes
    )
    model.load_state_dict(checkpoint["state_dict_ema"])
    model.eval().to(device)

    with open(class_map_path, "r") as f:
        class_map = {i: line.strip() for i, line in enumerate(f)}

    if len(class_map) != checkpoint["args"].num_classes:
        raise ValueError(
            f"class_map should have {checkpoint['args'].num_classes} entries, "
            f"got {len(class_map)}"
        )

    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform, class_map


def generate_cams(model, dataloader, target_layer, output_dir, device,
                   distance_threshold=0.05, gpu_batch=5):
    """Run ViT_CX on every image of the dataloader and save each CAM as .csv."""
    os.makedirs(output_dir, exist_ok=True)

    for images, filenames in tqdm(dataloader, desc="Generating CAMs"):
        for image, filename in zip(images, filenames):
            input_tensor = image.unsqueeze(0).to(device)

            result = ViT_CX(
                model,
                input_tensor,
                target_layer,
                target_category=None,
                distance_threshold=distance_threshold,
                gpu_batch=gpu_batch,
            )
            result = (result - np.min(result)) / (np.max(result) - np.min(result))

            out_path = os.path.join(output_dir, f"{filename}.csv")
            np.savetxt(out_path, result, delimiter=",", fmt="%.6f")

            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CAMs with DinoV2 + ViT_CX")
    parser.add_argument("--input_dir", required=True, help="Directory of input images")
    parser.add_argument("--output_dir", required=True, help="Directory to save CAM matrices (.csv)")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth.tar)")
    parser.add_argument("--class_map", required=True, help="Path to class_map.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--distance_threshold", type=float, default=0.05)
    parser.add_argument("--gpu_batch", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transform, class_map = load_model(args.checkpoint, args.class_map, device)
    target_layer = model.blocks[-1].norm1

    dataset = ImageDatasetWithFilename(args.input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    generate_cams(
        model,
        dataloader,
        target_layer,
        args.output_dir,
        device,
        distance_threshold=args.distance_threshold,
        gpu_batch=args.gpu_batch,
    )


if __name__ == "__main__":
    main()

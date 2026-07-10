"""
Train a DeepLabV3+ model for leaf disease symptom segmentation.

Corresponds to Section 2.3 ("Segmentation model training") of the
manuscript. Works identically whether the training masks come from
manual annotations (fully supervised) or from any of the pseudo-mask
pipelines (weakly supervised) -- only --masks_dir changes.

Reproducing the two training regimes from the manuscript:

    Fully supervised:
        python train_deeplabv3.py ... --lr 0.0001 --no-scheduler --augmentation

    Weakly supervised (any pseudo-mask pipeline):
        python train_deeplabv3.py ... --lr 0.1 --scheduler --no-augmentation

Usage
-----
python train_deeplabv3.py \
    --images_dir /path/to/images \
    --masks_dir /path/to/masks \
    --train_list /path/to/train_files.txt \
    --test_list /path/to/test_files.txt \
    --data_dir /path/to/working_dir \
    --save_dir /path/to/save/model \
    --epochs 150 --lr 0.1
"""

import argparse
import os
import shutil

import albumentations as album
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils.metrics
import torch
from PIL import Image, ExifTags
from skimage import io
from torch.utils.data import DataLoader


CLASS_NAMES = ["background", "infection"]
CLASS_RGB_VALUES = [[0, 0, 0], [255, 255, 255]]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def clear_folder(folder_path):
    """Empty a folder without deleting it, so the train/test structure can
    be reused across pipelines (only the content changes)."""
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)


def one_hot_encode(label, label_values):
    semantic_map = [np.all(np.equal(label, colour), axis=-1) for colour in label_values]
    return np.stack(semantic_map, axis=-1)


def reverse_one_hot(image):
    return np.argmax(image, axis=-1)


def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    return colour_codes[image.astype(int)]


def correct_orientation(image, image_path):
    """Rotate a mask/image array according to the EXIF orientation of the
    source photo (smartphone images may store orientation as metadata
    rather than actually rotating pixels)."""
    img = Image.open(image_path)
    exif = img.getexif()
    orientation_tag = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
    orientation = exif.get(orientation_tag, None)

    if orientation == 3:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif orientation == 8:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif orientation == 6:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(preprocessing_fn=None, target_size=(518, 518)):
    transforms = [album.Resize(target_size[0], target_size[1])]
    if preprocessing_fn:
        transforms.append(album.Lambda(image=preprocessing_fn))
    transforms.append(album.Lambda(image=to_tensor, mask=to_tensor))
    return album.Compose(transforms, is_check_shapes=True)


TRAIN_AUGMENTATION = album.Compose([
    album.HorizontalFlip(p=0.5),
    album.VerticalFlip(p=0.5),
    album.RandomRotate90(p=0.5),
    album.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, shift_limit=0.1, p=0.5),
    album.RandomBrightnessContrast(p=0.5),
])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class LeafSymptomDataset(torch.utils.data.Dataset):
    """Loads an image and its binary symptom mask, with optional EXIF
    correction and augmentation."""

    def __init__(self, images_dir, masks_dir, class_rgb_values=None,
                 preprocessing=None, correction=False, augmentation=None,
                 image_size=(518, 518), return_filename=True):
        self.image_paths = [os.path.join(images_dir, f) for f in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, f) for f in sorted(os.listdir(masks_dir))]
        self.class_rgb_values = class_rgb_values
        self.preprocessing = preprocessing
        self.correction = correction
        self.augmentation = augmentation
        self.image_size = image_size
        self.return_filename = return_filename

    def __getitem__(self, i):
        image = io.imread(self.image_paths[i])
        image = cv2.resize(image, self.image_size)

        mask = io.imread(self.mask_paths[i])
        if mask.dtype == bool:
            mask = mask.astype("uint8") * 255
        mask = cv2.resize(mask, self.image_size)

        if self.correction:
            mask = correct_orientation(mask, self.image_paths[i])

        if len(np.unique(mask)) != 2:
            mask[mask < 128] = 0
            mask[mask >= 128] = 1

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        mask_rgb = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2RGB)
        mask = one_hot_encode(mask_rgb, self.class_rgb_values).astype("float")

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        filename = os.path.basename(self.image_paths[i])
        if self.return_filename:
            return image, mask, filename
        return image, mask

    def __len__(self):
        return len(self.image_paths)


# --------------------------------------------------------------------------- #
# Data preparation (train/test split materialization)
# --------------------------------------------------------------------------- #

def prepare_split_folders(images_dir, masks_dir, train_list, test_list,
                           data_dir, masks_dir_test=None):
    """Copy images/masks into data_dir/{train,test}/{images,gt} according to
    the given file lists (one filename per line)."""
    x_train_dir = os.path.join(data_dir, "train/images")
    y_train_dir = os.path.join(data_dir, "train/gt")
    x_test_dir = os.path.join(data_dir, "test/images")
    y_test_dir = os.path.join(data_dir, "test/gt")

    for d in [x_train_dir, y_train_dir, x_test_dir, y_test_dir]:
        os.makedirs(d, exist_ok=True)
        clear_folder(d)

    with open(train_list, "r") as f:
        train_files = [line.strip() for line in f]
    with open(test_list, "r") as f:
        test_files = [line.strip() for line in f]

    def _copy(files, src_gt_dir, dst_im_dir, dst_gt_dir):
        n = 0
        for file_name in files:
            src_im = os.path.join(images_dir, file_name)
            src_gt = os.path.join(src_gt_dir, file_name)
            if os.path.exists(src_im) and os.path.exists(src_gt):
                shutil.copy(src_im, os.path.join(dst_im_dir, file_name))
                shutil.copy(src_gt, os.path.join(dst_gt_dir, file_name))
                n += 1
        return n

    n_train = _copy(train_files, masks_dir, x_train_dir, y_train_dir)
    n_test = _copy(test_files, masks_dir_test or masks_dir, x_test_dir, y_test_dir)

    print(f"Train images: {n_train} / {len(train_files)}")
    print(f"Test images: {n_test} / {len(test_files)}")

    return x_train_dir, y_train_dir, x_test_dir, y_test_dir


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def build_model(encoder="resnet101", encoder_weights="imagenet"):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=len(CLASS_NAMES),
        activation="softmax2d",
    )
    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
    return model, preprocessing_fn


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(args):
    os.makedirs(os.path.dirname(args.save_dir) or ".", exist_ok=True)

    x_train_dir, y_train_dir, x_test_dir, y_test_dir = prepare_split_folders(
        args.images_dir, args.masks_dir, args.train_list, args.test_list,
        args.data_dir, masks_dir_test=args.masks_dir_test,
    )

    model, preprocessing_fn = build_model(args.encoder, args.encoder_weights)

    augmentation = TRAIN_AUGMENTATION if args.augmentation else None
    train_dataset = LeafSymptomDataset(
        x_train_dir, y_train_dir,
        preprocessing=get_preprocessing(preprocessing_fn, (args.image_size, args.image_size)),
        class_rgb_values=CLASS_RGB_VALUES,
        augmentation=augmentation,
        correction=args.exif_correction,
        return_filename=False,
    )
    valid_dataset = LeafSymptomDataset(
        x_test_dir, y_test_dir,
        preprocessing=get_preprocessing(preprocessing_fn, (args.image_size, args.image_size)),
        class_rgb_values=CLASS_RGB_VALUES,
        correction=args.exif_correction,
        return_filename=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loss = smp.utils.losses.DiceLoss(ignore_channels=[0])
    metrics = [smp.utils.metrics.IoU(ignore_channels=[0])]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.lr)])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-6,
    )

    train_epoch = smp.utils.train.TrainEpoch(
        model, loss=loss, metrics=metrics, optimizer=optimizer, device=device, verbose=True,
    )
    valid_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=device, verbose=True,
    )

    best_loss = float("inf")
    history = {"train_losses": [], "train_ious": [], "valid_losses": [], "valid_ious": []}
    save_path = f"{args.save_dir}_best_loss.pth"

    for epoch in range(args.epochs):
        print(f"\nEpoch: {epoch}")
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        history["train_losses"].append(train_logs["dice_loss"])
        history["train_ious"].append(train_logs["iou_score"])
        history["valid_losses"].append(valid_logs["dice_loss"])
        history["valid_ious"].append(valid_logs["iou_score"])

        if valid_logs["dice_loss"] < best_loss:
            best_loss = valid_logs["dice_loss"]

        torch.save({
            "model_state_dict": model.state_dict(),
            "iou_score": valid_logs["iou_score"],
            "loss": valid_logs["dice_loss"],
            **history,
        }, save_path)
        print(f"Model saved to {save_path} (val loss = {valid_logs['dice_loss']:.4f})")

        if args.scheduler:
            scheduler.step(valid_logs["dice_loss"])

    print(f"Training complete. Best model saved to {save_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ for leaf symptom segmentation")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--masks_dir", required=True, help="Training masks (manual GT or pseudo-masks)")
    parser.add_argument("--masks_dir_test", default=None, help="Test masks, if different from --masks_dir")
    parser.add_argument("--train_list", required=True, help=".txt file with one train filename per line")
    parser.add_argument("--test_list", required=True, help=".txt file with one test filename per line")
    parser.add_argument("--data_dir", required=True, help="Working dir where train/test splits are materialized")
    parser.add_argument("--save_dir", required=True, help="Path prefix used to save model checkpoints")

    parser.add_argument("--encoder", default="resnet101")
    parser.add_argument("--encoder_weights", default="imagenet")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--scheduler", action=argparse.BooleanOptionalAction, default=True,
                         help="Reduce-on-plateau LR schedule (weakly supervised models)")
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=False,
                         help="Train-time augmentation (used only for the fully-supervised model)")
    parser.add_argument("--exif_correction", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

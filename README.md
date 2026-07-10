# Text-Guided Leaf Symptom Segmentation

Code associated with the manuscript:

> **Text guidance is powerful but prompt-sensitive for weakly-supervised leaf symptom segmentation**
> Romane Dubois, Lydia Bousset, Stéphane Jumel, Melen Leclerc, Nicolas Parisey, Alexis Joly

This repository implements and compares three pseudo-mask generation strategies for weakly supervised plant disease symptom segmentation — CAM-based refinement (SAM / SAM3), text-guided segmentation (SAM3), and a hybrid box+text approach — used to train a DeepLabV3+ segmentation model without pixel-level annotations.

## Repository structure

```
src/
├── pseudo_masks/
│   ├── cam_generation.py      # Generate CAM matrices from DINOv2 model
│   ├── binarize_cam.py        # Binarize CAM matrices into seed masks
│   ├── sam_refine.py          # CAM + SAM(box) refinement
│   ├── sam3_refine.py         # CAM + SAM3(box) and CAM + SAM3(box+txt) refinement
│   ├── sam3_text_only.py      # SAM3(txt) — text-prompt-only pseudo-masks
│   └── filter_empty_masks.py  # Discard images with no predicted mask
├── training/
│   └── train_deeplabv3.py     # Train DeepLabV3+ (fully or weakly supervised)
└── evaluation/
    └── predict_deeplabv3.py   # Run inference with a trained checkpoint
```

## Data availability

No raw image data is stored in this repository. The datasets used in the manuscript are publicly available:

- **Oilseed rape leaf disease dataset** (2,540 images, 7 diseases, 601 pixel-level annotations, 4 annotators): [https://doi.org/10.57745/R3HZOQ](https://doi.org/10.57745/R3HZOQ)
- **Apple leaf disease dataset** (865 images, 5 diseases, used to assess generalization): [https://data.mendeley.com/datasets/tsfxgsp3z6](https://data.mendeley.com/datasets/tsfxgsp3z6)

## Installation

```bash
git clone https://github.com/Romanedbs/Text-Guided-Leaf-Symptom-Segmentation.git
cd Text-Guided-Leaf-Symptom-Segmentation
```

This project additionally relies on three models installed directly from their official repositories (not on PyPI):

```bash
# SAM (original Segment Anything Model)
pip install git+https://github.com/facebookresearch/segment-anything.git

# SAM3
pip install git+https://github.com/facebookresearch/sam3.git

# ViT_CX (CAM method used for the DinoV2 classifier)
pip install git+https://github.com/vaynexie/CausalX-ViT.git
```

Model checkpoints (SAM `sam_vit_h_4b8939.pth`, SAM3 weights, and the DinoV2 classifier checkpoint) must be downloaded separately following each project's instructions and are not included in this repository.

All experiments were run with Python 3.12.9, PyTorch 2.7.0, and CUDA 12.6.

## Usage example

Example pipeline for the hybrid **CAM + SAM3(box+txt)** strategy with **"spots"** as text prompt, from binary seed masks to a trained model:

```bash
# 1. Binarize CAM matrices into seed masks
python src/pseudo_masks/binarize_cam.py \
    --cam_dir data/cams \
    --output_dir data/seed_masks \
    --method percentile --percentile_top 1

# 2. Refine seed masks with SAM3, combining the box prompt with a text prompt
python src/pseudo_masks/sam3_refine.py \
    --image_dir data/images \
    --seed_mask_dir data/seed_masks \
    --output_dir data/pseudo_masks/box_txt_spots \
    --text_prompt "spots"

# 3. Discard images for which no mask was predicted
python src/pseudo_masks/filter_empty_masks.py \
    --input_dir data/pseudo_masks/box_txt_spots \
    --output_dir data/pseudo_masks/box_txt_spots_filtered

# 4. Train DeepLabV3+ on the resulting pseudo-masks (weakly supervised regime)
python src/training/train_deeplabv3.py \
    --images_dir data/images \
    --masks_dir data/pseudo_masks/box_txt_spots_filtered \
    --train_list data/splits/train.txt \
    --test_list data/splits/test.txt \
    --data_dir data/working_dir \
    --save_dir models/cam_sam3_boxtxt_spots \
    --lr 0.1 --scheduler --no-augmentation

# 5. Predict on the test set with the trained checkpoint
python src/evaluation/predict_deeplabv3.py \
    --checkpoint models/cam_sam3_boxtxt_spots_best_loss.pth \
    --images_dir data/working_dir/test/images \
    --masks_dir data/working_dir/test/gt \
    --output_dir predictions/cam_sam3_boxtxt_spots
```

For the text-only **SAM3(txt)** pipeline, use `src/pseudo_masks/sam3_text_only.py` instead of steps 1–2 (no seed mask needed). For the fully supervised baseline, run `train_deeplabv3.py` directly on manual annotations with `--lr 0.0001 --no-scheduler --augmentation`.

## Citation

This work is currently available as a preprint on bioRxiv. Please cite:

```bibtex
@article{dubois2026textguided,
  title   = {Text guidance is powerful but prompt-sensitive for weakly-supervised leaf symptom segmentation},
  author  = {Dubois, Romane and Bousset, Lydia and Jumel, St{\'e}phane and Leclerc, Melen and Parisey, Nicolas and Joly, Alexis},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {[add bioRxiv DOI here]}
}
```

> This citation will be updated with the peer-reviewed journal reference once the manuscript is accepted for publication.

## Acknowledgments

This work was supported by the French National Research Agency (ANR) under grant Pl@ntAgroEco (ANR-22-PEAE-0009).

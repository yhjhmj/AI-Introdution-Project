# Mask-Constrained CycleGAN

Instance-level unpaired translation (horse <-> zebra) with background freeze.

## Pipeline
1. Stage 1: vanilla CycleGAN
2. Stage 2: Mask R-CNN + soft mask compose + background L1
3. Stage 3: PSNR / SSIM / L1 ablation and vis

## Layout
```
mask_cyclegan/
  configs/default.yaml
  data/dataset.py
  models/
  mask/
  losses/
  metrics/
  scripts/
```

## Data
```
datasets/horse2zebra/trainA trainB testA testB
```

## Run
```
pip install -r requirements.txt
python scripts/extract_masks.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml --stage 1
python scripts/train.py --config configs/default.yaml --stage 2
python scripts/test.py --config configs/default.yaml --ckpt checkpoints/latest.pth
python scripts/eval.py --config configs/default.yaml --pred_dir results/test
```

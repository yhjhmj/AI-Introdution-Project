import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import IMG_EXTS
from mask.extractor import MaskExtractor
from utils.io import load_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    extractor = MaskExtractor(device=device)
    tf = transforms.ToTensor()
    src_root = Path(cfg["data_root"])
    dst_root = Path(cfg["mask_root"])
    for split in ["trainA", "trainB", "testA", "testB"]:
        src = src_root / split
        dst = dst_root / split
        dst.mkdir(parents=True, exist_ok=True)
        if not src.exists():
            continue
        paths = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS]
        for p in tqdm(paths, desc=split):
            img = tf(Image.open(p).convert("RGB")).unsqueeze(0)
            mask = extractor(img.to(device)).squeeze().cpu().numpy()
            Image.fromarray((mask * 255).astype("uint8")).save(dst / (p.stem + ".png"))


if __name__ == "__main__":
    main()

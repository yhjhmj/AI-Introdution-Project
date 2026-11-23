import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metrics.eval import background_metrics
from utils.io import load_yaml


def load_rgb(path, size):
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--pred_dir", required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    size = cfg["image_size"]
    src_dir = Path(cfg["data_root"]) / "testA"
    mask_dir = Path(cfg["mask_root"]) / "testA"
    pred_dir = Path(args.pred_dir)
    scores = []
    for pred in sorted(pred_dir.glob("*.png")):
        src = src_dir / pred.name
        mask = mask_dir / pred.name
        if not src.exists() or not mask.exists():
            continue
        m = np.asarray(Image.open(mask).convert("L").resize((size, size))) / 255.0
        scores.append(background_metrics(load_rgb(src, size), load_rgb(pred, size), m))
    if not scores:
        print("no matched files")
        return
    mean = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    print(json.dumps(mean, indent=2))


if __name__ == "__main__":
    main()

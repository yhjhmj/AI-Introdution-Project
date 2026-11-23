import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import UnpairedMaskDataset
from mask.compose import compose, soften_mask
from models.cyclegan import CycleGAN
from utils.io import load_yaml, save_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--stage", type=int, default=2)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    ds = UnpairedMaskDataset(cfg["data_root"], cfg.get("mask_root"), "test", cfg["image_size"])
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    net = CycleGAN().to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device)["net"])
    net.eval()
    out_dir = Path(cfg["result_dir"]) / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if "A" not in batch:
                continue
            real_a = batch["A"].to(device)
            fake = net.G_AB(real_a)
            if args.stage >= 2 and "A_mask" in batch:
                mask = soften_mask(batch["A_mask"].to(device), cfg["soft_mask_blur"])
                fake = compose(real_a, fake, mask)
            save_image(fake, out_dir / f"{i:05d}.png")


if __name__ == "__main__":
    main()

import argparse
import itertools
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.dataset import UnpairedMaskDataset
from losses.objectives import GANLoss, background_l1, cycle_l1
from mask.compose import compose, soften_mask
from models.cyclegan import CycleGAN
from utils.io import load_yaml, set_seed


def ones_like_mask(img):
    return torch.ones(img.size(0), 1, img.size(2), img.size(3), device=img.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    ds = UnpairedMaskDataset(cfg["data_root"], cfg.get("mask_root"), "train", cfg["image_size"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"])
    net = CycleGAN().to(device)
    gan_loss = GANLoss()
    opt_g = torch.optim.Adam(
        itertools.chain(net.G_AB.parameters(), net.G_BA.parameters()),
        lr=cfg["lr"], betas=(cfg["beta1"], 0.999),
    )
    opt_d = torch.optim.Adam(
        itertools.chain(net.D_A.parameters(), net.D_B.parameters()),
        lr=cfg["lr"], betas=(cfg["beta1"], 0.999),
    )
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        pbar = tqdm(loader, desc=f"stage{args.stage} epoch {epoch}")
        for batch in pbar:
            if "A" not in batch or "B" not in batch:
                continue
            real_a = batch["A"].to(device)
            real_b = batch["B"].to(device)
            mask_a = batch.get("A_mask")
            mask_b = batch.get("B_mask")
            mask_a = mask_a.to(device) if mask_a is not None else ones_like_mask(real_a)
            mask_b = mask_b.to(device) if mask_b is not None else ones_like_mask(real_b)
            mask_a = soften_mask(mask_a, cfg["soft_mask_blur"])
            mask_b = soften_mask(mask_b, cfg["soft_mask_blur"])
            fake_b_raw = net.G_AB(real_a)
            fake_a_raw = net.G_BA(real_b)
            if args.stage >= 2:
                fake_b = compose(real_a, fake_b_raw, mask_a)
                fake_a = compose(real_b, fake_a_raw, mask_b)
            else:
                fake_b, fake_a = fake_b_raw, fake_a_raw
            rec_a = net.G_BA(fake_b)
            rec_b = net.G_AB(fake_a)
            opt_g.zero_grad()
            loss_g = gan_loss(net.D_B(fake_b), True) + gan_loss(net.D_A(fake_a), True)
            loss_g = loss_g + cfg["lambda_cycle"] * (cycle_l1(rec_a, real_a) + cycle_l1(rec_b, real_b))
            loss_g = loss_g + cfg["lambda_identity"] * (
                cycle_l1(net.G_BA(real_a), real_a) + cycle_l1(net.G_AB(real_b), real_b)
            )
            if args.stage >= 2:
                loss_g = loss_g + cfg["lambda_bg"] * (
                    background_l1(fake_b, real_a, mask_a) + background_l1(fake_a, real_b, mask_b)
                )
            loss_g.backward()
            opt_g.step()
            opt_d.zero_grad()
            loss_d = gan_loss(net.D_A(real_a), True) + gan_loss(net.D_A(fake_a.detach()), False)
            loss_d = loss_d + gan_loss(net.D_B(real_b), True) + gan_loss(net.D_B(fake_b.detach()), False)
            loss_d.backward()
            opt_d.step()
            pbar.set_postfix(G=float(loss_g), D=float(loss_d))
        torch.save({"net": net.state_dict(), "epoch": epoch, "stage": args.stage}, ckpt_dir / "latest.pth")


if __name__ == "__main__":
    main()

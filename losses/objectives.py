import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def __call__(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.loss(pred, target)


def cycle_l1(recon, src):
    return torch.mean(torch.abs(recon - src))


def background_l1(out, src, mask):
    bg = 1.0 - mask
    return torch.mean(torch.abs((out - src) * bg))

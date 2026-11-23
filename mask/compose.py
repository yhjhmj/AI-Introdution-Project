import torch
import torch.nn.functional as F


def soften_mask(mask, ksize=5):
    if ksize <= 1:
        return mask
    pad = ksize // 2
    kernel = torch.ones(1, 1, ksize, ksize, device=mask.device) / (ksize * ksize)
    return F.conv2d(mask, kernel, padding=pad).clamp(0, 1)


def compose(src, fake, mask):
    return mask * fake + (1.0 - mask) * src

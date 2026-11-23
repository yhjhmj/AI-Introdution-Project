import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def background_metrics(src, pred, mask):
    bg = mask < 0.5
    if bg.sum() == 0:
        return {"psnr": 0.0, "ssim": 0.0, "l1": 0.0}
    l1 = float(np.mean(np.abs(src[bg] - pred[bg])))
    psnr = float(peak_signal_noise_ratio(src * bg[..., None], pred * bg[..., None], data_range=1.0))
    ssim = float(structural_similarity(src, pred, channel_axis=2, data_range=1.0, win_size=7))
    return {"psnr": psnr, "ssim": ssim, "l1": l1}

import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_to_numpy(t):
    t = (t.detach().cpu().clamp(-1, 1) + 1) / 2
    arr = t.squeeze(0).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def save_image(t, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(tensor_to_numpy(t)).save(path)

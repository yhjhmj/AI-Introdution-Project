from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


class UnpairedMaskDataset(Dataset):
    def __init__(self, root, mask_root=None, split="train", size=128):
        self.a_paths = list_images(Path(root) / f"{split}A")
        self.b_paths = list_images(Path(root) / f"{split}B")
        self.mask_root = Path(mask_root) if mask_root else None
        self.split = split
        self.tf = transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.mask_tf = transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor()]
        )
        self.len = max(len(self.a_paths), len(self.b_paths), 1)

    def _load_mask(self, img_path, domain):
        if self.mask_root is None:
            return None
        mask_path = self.mask_root / f"{self.split}{domain}" / (img_path.stem + ".png")
        if not mask_path.exists():
            return None
        return self.mask_tf(Image.open(mask_path).convert("L"))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample = {}
        if self.a_paths:
            a = self.a_paths[idx % len(self.a_paths)]
            sample["A"] = self.tf(Image.open(a).convert("RGB"))
            sample["A_path"] = str(a)
            m = self._load_mask(a, "A")
            if m is not None:
                sample["A_mask"] = m
        if self.b_paths:
            b = self.b_paths[idx % len(self.b_paths)]
            sample["B"] = self.tf(Image.open(b).convert("RGB"))
            sample["B_path"] = str(b)
            m = self._load_mask(b, "B")
            if m is not None:
                sample["B_mask"] = m
        return sample

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms.functional import convert_image_dtype

HORSE_LABEL = 18


class MaskExtractor:
    def __init__(self, device="cpu", score_thresh=0.5):
        self.device = device
        self.score_thresh = score_thresh
        self.model = maskrcnn_resnet50_fpn(weights="DEFAULT").to(device).eval()

    @torch.no_grad()
    def __call__(self, image_01):
        inp = convert_image_dtype(image_01.squeeze(0), torch.float).to(self.device)
        pred = self.model([inp])[0]
        keep = (pred["labels"] == HORSE_LABEL) & (pred["scores"] > self.score_thresh)
        h, w = image_01.shape[-2:]
        mask = torch.zeros(1, 1, h, w, device=image_01.device)
        if keep.any():
            inst = pred["masks"][keep]
            mask[0, 0] = inst.max(dim=0).values[0]
        return mask

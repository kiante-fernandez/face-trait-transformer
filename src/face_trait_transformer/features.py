"""Image preprocessing, device selection, and DINOv2 feature extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def pick_device() -> torch.device:
    """Select the best available torch device (cuda > mps > cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_transform(image_size: int = 224):
    """The DINOv2 preprocessing pipeline (torchvision v2 API).

    Bicubic resize (shorter side = image_size * 256/224), center-crop to
    image_size x image_size, convert to float tensor in [0, 1], ImageNet
    mean/std normalization.
    """
    from torchvision.transforms import v2  # lazy import keeps torchvision optional at import

    resize_to = int(round(image_size * 256 / 224))
    return v2.Compose([
        v2.Resize(resize_to, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
        v2.CenterCrop(image_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def load_dinov2(
    model_name: str = "dinov2_vitb14",
    device: torch.device | None = None,
    image_size: int = 224,
):
    """Load a DINOv2 backbone in eval mode along with its preprocessing transform.

    Returns ``(model, transform, device)``.
    """
    device = device or pick_device()
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model, build_transform(image_size=image_size), device


class ImageFolderList(torch.utils.data.Dataset):
    """Tiny Dataset wrapping a list of image paths and a preprocessing transform."""

    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img)


@torch.inference_mode()
def extract_cls(
    model: torch.nn.Module,
    transform,
    image_paths: list[Path],
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 2,
) -> np.ndarray:
    """Forward every image through the backbone, return CLS embeddings as (N, D)."""
    from tqdm import tqdm

    ds = ImageFolderList(image_paths, transform)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    feats: list[np.ndarray] = []
    for batch in tqdm(loader, desc="DINOv2 features"):
        batch = batch.to(device, non_blocking=True)
        out = model(batch)
        feats.append(out.detach().cpu().float().numpy())
    return np.concatenate(feats, axis=0)

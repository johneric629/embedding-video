"""DINOv2 / CLIP frame encoders."""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from embedding_clips.config import BACKBONES

IMAGENET = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

EncodeFn = Callable[[torch.Tensor], torch.Tensor]


def device_str(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(backbone: str, device: str) -> tuple[EncodeFn, int, int]:
    if backbone not in BACKBONES:
        raise ValueError(f"unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
    cfg = BACKBONES[backbone]

    if cfg["family"] == "dinov2":
        try:
            model = torch.hub.load("facebookresearch/dinov2", cfg["hub"], trust_repo=True)
        except Exception as e:
            raise SystemExit(
                "Failed to load DINOv2 via torch.hub (needs network once, or a warm "
                f"~/.cache/torch/hub). backbone={backbone!r}.\n{e}"
            ) from e
        model.eval().to(device)

        @torch.no_grad()
        def encode(batch_bchw: torch.Tensor) -> torch.Tensor:
            return model(batch_bchw)

        return encode, int(cfg["dim"]), int(cfg["size"])

    if cfg["family"] == "clip":
        try:
            import open_clip
        except ImportError as e:
            raise SystemExit(
                "OpenCLIP required for CLIP: pip install open_clip_torch"
            ) from e
        model, _, _ = open_clip.create_model_and_transforms(cfg["name"], pretrained=cfg["pretrained"])
        model.eval().to(device)

        @torch.no_grad()
        def encode(batch_bchw: torch.Tensor) -> torch.Tensor:
            return model.encode_image(batch_bchw)

        return encode, int(cfg["dim"]), int(cfg["size"])

    raise ValueError(backbone)


def preprocess_bgr(frames_bgr: list[np.ndarray], size: int, device: str) -> torch.Tensor:
    tensors = []
    for bgr in frames_bgr:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensors.append(IMAGENET(t))
    return torch.stack(tensors, 0).to(device)


def embed_frames(
    frames_bgr: list[np.ndarray],
    encode: EncodeFn,
    size: int,
    device: str,
) -> np.ndarray | None:
    """Mean-pool frame embeddings → L2-normalised (D,) float32."""
    if not frames_bgr:
        return None
    batch = preprocess_bgr(frames_bgr, size, device)
    with torch.no_grad():
        feats = encode(batch).float()
        if feats.ndim == 3:
            feats = feats.mean(1)
        vec = F.normalize(feats.mean(0, keepdim=True), dim=-1)[0]
    return vec.detach().cpu().numpy().astype(np.float32)

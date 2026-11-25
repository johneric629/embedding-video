"""Paths and backbone registry (portable; cwd + env, no host-specific paths)."""

from __future__ import annotations

import os
from pathlib import Path


def env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if not raw:
        return default
    return Path(raw).expanduser().resolve()


# Resolved at import: override with CRICKET_DATA / EMBEDDING_OUT, or pass --data/--out/--videos.
DEFAULT_DATA = env_path("CRICKET_DATA", Path.cwd() / "data" / "cricket_train")
DEFAULT_OUT = env_path("EMBEDDING_OUT", Path.cwd() / "out")

BACKBONES = {
    "dinov2_vits14": {"family": "dinov2", "hub": "dinov2_vits14", "dim": 384, "size": 224},
    "dinov2_vitb14": {"family": "dinov2", "hub": "dinov2_vitb14", "dim": 768, "size": 224},
    "dinov2_vitl14": {"family": "dinov2", "hub": "dinov2_vitl14", "dim": 1024, "size": 224},
    "clip_vitb32": {"family": "clip", "name": "ViT-B-32", "pretrained": "openai", "dim": 512, "size": 224},
    "clip_vitb16": {"family": "clip", "name": "ViT-B-16", "pretrained": "openai", "dim": 512, "size": 224},
}

DEFAULT_BACKBONE = "dinov2_vits14"
DEFAULT_FRAMES = 8

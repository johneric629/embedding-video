"""On-disk embedding store: embeddings.npy + ids.json + meta.json."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class StoreMeta:
    backbone: str
    dim: int
    frames: int
    image_size: int
    n_clips: int
    created_at: str
    source: str = "videos"
    shard: str | None = None


class EmbeddingStore:
    def __init__(self, root: Path, backbone: str):
        self.root = Path(root)
        self.backbone = backbone
        self.dir = self.root / backbone
        self.emb_path = self.dir / "embeddings.npy"
        self.ids_path = self.dir / "ids.json"
        self.meta_path = self.dir / "meta.json"

    def exists(self) -> bool:
        return self.emb_path.exists() and self.ids_path.exists()

    def load(self) -> tuple[np.ndarray, list[str], dict]:
        emb = np.load(self.emb_path)
        ids = json.loads(self.ids_path.read_text(encoding="utf-8"))
        meta = json.loads(self.meta_path.read_text(encoding="utf-8")) if self.meta_path.exists() else {}
        if len(ids) != len(emb):
            raise ValueError(f"ids ({len(ids)}) != embeddings ({len(emb)}) in {self.dir}")
        return emb, ids, meta

    def save(
        self,
        emb: np.ndarray,
        ids: list[str],
        *,
        frames: int,
        image_size: int,
        source: str = "videos",
        shard: str | None = None,
    ) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        np.save(self.emb_path, emb)
        self.ids_path.write_text(json.dumps(ids), encoding="utf-8")
        meta = StoreMeta(
            backbone=self.backbone,
            dim=int(emb.shape[1]),
            frames=frames,
            image_size=image_size,
            n_clips=len(ids),
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            source=source,
            shard=shard,
        )
        self.meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    def topk(self, query: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        emb, ids, _ = self.load()
        q = np.asarray(query, np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-8)
        scores = emb @ q
        k = min(k, len(ids))
        idx = np.argsort(-scores)[:k]
        return [(ids[int(i)], float(scores[int(i)])) for i in idx]


def merge_stores(part_dirs: list[Path], out: EmbeddingStore, *, frames: int, image_size: int) -> int:
    """Concatenate shard stores; skip duplicate ids (first wins)."""
    all_ids: list[str] = []
    vectors: list[np.ndarray] = []
    seen: set[str] = set()
    for d in part_dirs:
        emb = np.load(d / "embeddings.npy")
        ids = json.loads((d / "ids.json").read_text(encoding="utf-8"))
        for i, sid in enumerate(ids):
            if sid in seen:
                continue
            seen.add(sid)
            all_ids.append(sid)
            vectors.append(emb[i])
    if not vectors:
        raise SystemExit("nothing to merge")
    mat = np.stack(vectors, 0)
    out.save(mat, all_ids, frames=frames, image_size=image_size, source="merge")
    return len(all_ids)

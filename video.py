"""Frame sampling and clip path discovery."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def sample_frames(path: Path | str, n: int) -> list[np.ndarray]:
    """Uniformly sample up to n BGR frames from an mp4."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        frames: list[np.ndarray] = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        if not frames:
            return []
        idx = np.linspace(0, len(frames) - 1, num=min(n, len(frames)), dtype=int)
        return [frames[int(i)] for i in idx]

    idx = np.linspace(0, max(total - 1, 0), num=min(n, total), dtype=int)
    want = {int(i) for i in idx}
    out: list[np.ndarray] = []
    i = 0
    while i < total and len(out) < len(want):
        if not cap.grab():
            break
        if i in want:
            ok, f = cap.retrieve()
            if ok:
                out.append(f)
        i += 1
    cap.release()
    return out


def list_videos(videos_dir: Path, glob: str = "*.mp4") -> list[Path]:
    if not videos_dir.is_dir():
        return []
    return sorted(videos_dir.glob(glob))


def resolve_clip_path(
    clip_id: str,
    *,
    data_dir: Path,
    video_url: str | None = None,
) -> Path | None:
    """Find a local mp4 for a training id (segments preferred, then videos/)."""
    stem = Path(clip_id).stem
    safe = clip_id.replace("/", "_").replace("\\", "_")
    candidates = [
        data_dir / "segments" / f"{stem}.mp4",
        data_dir / "segments" / f"{safe}.mp4",
        data_dir / "videos" / f"{stem}.mp4",
        data_dir / "videos" / f"{safe}.mp4",
    ]
    if video_url:
        candidates.insert(0, data_dir / "videos" / f"{Path(video_url).stem}.mp4")

    man = data_dir / "segments" / "manifest.jsonl"
    if man.exists():
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("id") != clip_id and rec.get("clip") != clip_id:
                continue
            for key in ("segment", "path", "out"):
                raw = rec.get(key)
                if not raw:
                    continue
                p = Path(raw)
                if not p.is_absolute():
                    p = data_dir / "segments" / p.name
                candidates.insert(0, p)
            break

    for p in candidates:
        if p.exists():
            return p
    return None


def load_label_index(data_dir: Path) -> list[dict]:
    """Rows from Score labels.json: {id, video_url, gt, split?}."""
    path = data_dir / "labels.json"
    if not path.exists():
        return []
    labels = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for k, v in labels.items():
        if not isinstance(v, dict):
            continue
        rows.append({
            "id": k,
            "video_url": v.get("video_url"),
            "gt": v.get("groundtruth") or v.get("gt") or {},
        })
    # stable order by numeric train index when possible
    def sort_key(r: dict) -> tuple:
        s = Path(r["id"]).stem
        digits = "".join(ch for ch in s if ch.isdigit())
        return (0, int(digits)) if digits else (1, r["id"])

    rows.sort(key=sort_key)
    return rows


def iter_build_items(
    *,
    data_dir: Path | None,
    videos_dir: Path | None,
    limit: int = 0,
    shard_i: int = 0,
    shard_n: int = 1,
) -> list[tuple[str, Path]]:
    """Return (id, path) pairs to embed."""
    items: list[tuple[str, Path]] = []
    if data_dir and (data_dir / "labels.json").exists():
        for row in load_label_index(data_dir):
            path = resolve_clip_path(row["id"], data_dir=data_dir, video_url=row.get("video_url"))
            if path is not None:
                items.append((row["id"], path))
    elif videos_dir:
        for p in list_videos(videos_dir):
            items.append((p.stem, p))
    else:
        return []

    if shard_n > 1:
        items = [it for i, it in enumerate(items) if i % shard_n == shard_i]
    if limit > 0:
        items = items[:limit]
    return items

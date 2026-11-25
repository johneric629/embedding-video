"""CLI commands for the embedding_clips project."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from embedding_clips.config import (
    BACKBONES,
    DEFAULT_BACKBONE,
    DEFAULT_DATA,
    DEFAULT_FRAMES,
    DEFAULT_OUT,
)
from embedding_clips.encoder import device_str, embed_frames, load_encoder
from embedding_clips.store import EmbeddingStore, merge_stores
from embedding_clips.video import iter_build_items, sample_frames


def cmd_estimate(args: argparse.Namespace) -> None:
    n = args.n_clips
    if n <= 0:
        items = iter_build_items(data_dir=args.data, videos_dir=args.videos, limit=0)
        n = len(items) if items else 13824
    frames = args.frames
    cfg = BACKBONES[args.backbone]
    ms_gpu = {
        "dinov2_vits14": 5.5,
        "dinov2_vitb14": 9.2,
        "dinov2_vitl14": 33.0,
        "clip_vitb32": 6.0,
        "clip_vitb16": 12.0,
    }[args.backbone]
    ms_cpu = {
        "dinov2_vits14": 45.0,
        "dinov2_vitb14": 145.0,
        "dinov2_vitl14": 520.0,
        "clip_vitb32": 40.0,
        "clip_vitb16": 80.0,
    }[args.backbone]
    decode_ms = 120.0

    def hours(s: float) -> float:
        return n * s / 3600.0

    gpu_e2e = hours(decode_ms / 1000.0 + frames * ms_gpu / 1000.0)
    cpu_e2e = hours(decode_ms / 1000.0 + frames * ms_cpu / 1000.0)
    store_mb = n * cfg["dim"] * 4 / 1e6
    print(f"clips≈{n}  backbone={args.backbone}  dim={cfg['dim']}  frames={frames}")
    print(f"storage ≈ {store_mb:.1f} MB")
    print(f"GPU end-to-end ≈ {gpu_e2e:.2f} h   (T4 / similar)")
    print(f"CPU end-to-end ≈ {cpu_e2e:.2f} h")
    print("Tip: use delivery segments, not full 30s broadcasts.")

    if args.bench:
        import torch

        device = device_str(args.device)
        encode, dim, size = load_encoder(args.backbone, device)
        dummy = [np.zeros((360, 640, 3), np.uint8) for _ in range(frames)]
        from embedding_clips.encoder import preprocess_bgr

        batch = preprocess_bgr(dummy, size, device)
        for _ in range(5):
            encode(batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        reps = 20
        for _ in range(reps):
            encode(batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / reps / frames * 1000
        print(f"measured {ms:.1f} ms/frame on {device} → ≈ {hours(decode_ms/1000 + frames*ms/1000):.2f} h full")


def cmd_build(args: argparse.Namespace) -> None:
    device = device_str(args.device)
    encode, dim, size = load_encoder(args.backbone, device)
    items = iter_build_items(
        data_dir=args.data,
        videos_dir=args.videos,
        limit=args.limit,
        shard_i=args.shard_i,
        shard_n=args.shard_n,
    )
    if not items:
        raise SystemExit(
            "no videos found. Pass --data DIR (labels.json + segments/ or videos/) "
            "or --videos DIR of mp4s. Delivery segments preferred over full broadcasts."
        )

    out_root = args.out
    name = args.backbone
    if args.shard_n > 1:
        name = f"{args.backbone}_shard{args.shard_i}of{args.shard_n}"
    store = EmbeddingStore(out_root, name)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    done: set[str] = set()
    if store.exists() and not args.fresh:
        emb, ids, _ = store.load()
        vectors = [emb[i] for i in range(len(ids))]
        done = set(ids)
        print(f"resuming {len(done)} already in {store.dir}")

    t0 = time.perf_counter()
    n_ok = n_miss = 0
    sources: set[str] = set()
    for k, (cid, path) in enumerate(items):
        if cid in done:
            continue
        frames = sample_frames(path, args.frames)
        vec = embed_frames(frames, encode, size, device)
        if vec is None:
            n_miss += 1
            continue
        sources.add("segments" if "segments" in path.as_posix() else "videos")
        ids.append(cid)
        vectors.append(vec)
        done.add(cid)
        n_ok += 1
        if n_ok % 25 == 0 or (k + 1) == len(items):
            rate = n_ok / max(time.perf_counter() - t0, 1e-6)
            print(f"  {n_ok} new  {k+1}/{len(items)}  {rate:.2f} clips/s  miss={n_miss}")
            store.save(
                np.stack(vectors, 0),
                ids,
                frames=args.frames,
                image_size=size,
                source="+".join(sorted(sources)) or "videos",
                shard=f"{args.shard_i}/{args.shard_n}" if args.shard_n > 1 else None,
            )

    if not vectors:
        raise SystemExit(f"no embeddings written (miss={n_miss})")
    store.save(
        np.stack(vectors, 0),
        ids,
        frames=args.frames,
        image_size=size,
        source="+".join(sorted(sources)) or "videos",
        shard=f"{args.shard_i}/{args.shard_n}" if args.shard_n > 1 else None,
    )
    print(f"done → {store.dir}  n={len(ids)}  ({(time.perf_counter()-t0)/60:.1f} min)")


def cmd_query(args: argparse.Namespace) -> None:
    device = device_str(args.device)
    encode, dim, size = load_encoder(args.backbone, device)
    store = EmbeddingStore(args.out, args.backbone)
    if not store.exists():
        raise SystemExit(f"no store at {store.dir}")

    path = Path(args.video) if args.video else None
    if args.url:
        import tempfile
        import urllib.request

        tmp = Path(tempfile.gettempdir()) / "embedding_clips_query.mp4"
        print(f"download {args.url}")
        with urllib.request.urlopen(args.url, timeout=120) as resp:
            tmp.write_bytes(resp.read())
        path = tmp
    if path is None or not path.exists():
        raise SystemExit("pass a local video path or --url")

    frames = sample_frames(path, args.frames)
    q = embed_frames(frames, encode, size, device)
    if q is None:
        raise SystemExit("could not read frames")
    hits = store.topk(q, args.topk)

    gt_by_id = {}
    if args.data and (args.data / "labels.json").exists():
        from embedding_clips.video import load_label_index

        gt_by_id = {r["id"]: r["gt"] for r in load_label_index(args.data)}

    print(f"query={path.name}  store={store.dir}  topk={args.topk}")
    print(f"{'rank':>4}  {'cos':>7}  id")
    for rank, (sid, cos) in enumerate(hits, 1):
        print(f"{rank:4d}  {cos:7.4f}  {sid}")
        if args.show_gt and sid in gt_by_id:
            gt = gt_by_id[sid]
            keys = ("kph", "bounce_x", "stump_y", "stump_z", "release_z")
            print("       gt " + " ".join(f"{k}={gt.get(k)}" for k in keys))


def cmd_merge(args: argparse.Namespace) -> None:
    parts = [Path(p) for p in args.parts]
    for p in parts:
        if not (p / "embeddings.npy").exists():
            raise SystemExit(f"missing embeddings in {p}")
    out = EmbeddingStore(args.out, args.backbone)
    n = merge_stores(parts, out, frames=args.frames, image_size=224)
    print(f"merged {n} → {out.dir}")


def cmd_validate(args: argparse.Namespace) -> None:
    store = EmbeddingStore(args.out, args.backbone)
    emb, ids, meta = store.load()
    # norms
    norms = np.linalg.norm(emb, axis=1)
    print(f"n={len(ids)} dim={emb.shape[1]} meta={meta}")
    print(f"L2 norms: mean={norms.mean():.4f} min={norms.min():.4f} max={norms.max():.4f}")
    # self-retrieval on a sample
    sample = min(args.sample, len(ids))
    idx = np.linspace(0, len(ids) - 1, num=sample, dtype=int)
    ok = 0
    for i in idx:
        scores = emb @ emb[i]
        scores[i] = -1  # exclude self for "second" check; for true self use argmax including i
        # true self-retrieval: argmax should be i
        scores_full = emb @ emb[i]
        if int(np.argmax(scores_full)) == int(i):
            ok += 1
    print(f"self-retrieval @1: {ok}/{sample} ({100*ok/sample:.1f}%)")


def cmd_near_dups(args: argparse.Namespace) -> None:
    store = EmbeddingStore(args.out, args.backbone)
    emb, ids, _ = store.load()
    thr = args.threshold
    # for each vector, find neighbours above thr (excluding self)
    pairs = []
    for i in range(len(ids)):
        scores = emb @ emb[i]
        scores[i] = -1
        j = int(np.argmax(scores))
        if scores[j] >= thr:
            a, b = ids[i], ids[j]
            if a > b:
                a, b = b, a
            pairs.append((float(scores[j]), a, b))
    # unique undirected
    seen = set()
    uniq = []
    for cos, a, b in sorted(pairs, reverse=True):
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((cos, a, b))
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"cos": c, "a": a, "b": b}) for c, a, b in uniq]
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"{len(uniq)} near-dup pairs (cos≥{thr}) → {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="embedding_clips",
        description="DINOv2/CLIP similarity store for cricket delivery clips",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
        sp.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
        sp.add_argument("--device", default=None)
        sp.add_argument("--out", type=Path, default=DEFAULT_OUT, help="store root (default embedding_clips/out)")
        sp.add_argument("--data", type=Path, default=DEFAULT_DATA, help="cricket_train data dir")
        sp.add_argument("--videos", type=Path, default=None, help="flat directory of mp4s (optional)")

    e = sub.add_parser("estimate", help="time / storage estimate")
    add_common(e)
    e.add_argument("--n-clips", type=int, default=0)
    e.add_argument("--bench", action="store_true")

    b = sub.add_parser("build", help="embed local clips into the store")
    add_common(b)
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--fresh", action="store_true")
    b.add_argument("--shard-i", type=int, default=0)
    b.add_argument("--shard-n", type=int, default=1)

    q = sub.add_parser("query", help="nearest neighbours for one video")
    add_common(q)
    q.add_argument("video", nargs="?", default=None)
    q.add_argument("--url", default=None)
    q.add_argument("--topk", type=int, default=10)
    q.add_argument("--show-gt", action="store_true")

    m = sub.add_parser("merge", help="merge shard stores")
    m.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    m.add_argument("--out", type=Path, default=DEFAULT_OUT)
    m.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    m.add_argument("parts", nargs="+", help="shard directories containing embeddings.npy")

    v = sub.add_parser("validate", help="self-retrieval sanity check")
    add_common(v)
    v.add_argument("--sample", type=int, default=200)

    d = sub.add_parser("near-dups", help="list pairs above cosine threshold")
    add_common(d)
    d.add_argument("--threshold", type=float, default=0.92)
    d.add_argument("--report", type=Path, default=None)

    return p


def main(argv: list[str] | None = None) -> None:
    p = build_parser()
    args = p.parse_args(argv)
    if getattr(args, "report", None) is None and args.cmd == "near-dups":
        args.report = Path(args.out) / args.backbone / "near_dups.jsonl"
    if args.cmd == "estimate":
        cmd_estimate(args)
    elif args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "query":
        cmd_query(args)
    elif args.cmd == "merge":
        cmd_merge(args)
    elif args.cmd == "validate":
        cmd_validate(args)
    elif args.cmd == "near-dups":
        cmd_near_dups(args)
    else:
        raise SystemExit(args.cmd)

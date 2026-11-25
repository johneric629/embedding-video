# embedding-clips

DINOv2 / CLIP **video similarity store** for cricket delivery clips.

- Embed delivery segments (or any folder of mp4s)
- **8 frames @ 224**, mean-pool, L2-normalise
- Default backbone: **DINOv2 ViT-S/14** (384-d)
- Cosine search (brute-force; ~14k vectors is tiny)
- Shard / merge for multi-GPU (e.g. Kaggle T4×2)

This package is **self-contained**. It does **not** include a scene/delivery finder — cut delivery windows first (or pass already-segmented mp4s), then embed.

Similarity is for dedup / domain / neighbour **priors**. Do not copy neighbour GT as a live scoring answer.

## Requirements

- Python **3.10+**
- Network on first DINOv2 run (`torch.hub` pulls weights)
- GPU optional (CUDA torch wheel recommended for full builds)

## Install

**Standalone repo:** clone this project, then `cd` into it (the folder that contains `pyproject.toml`).  
**Monorepo:** `cd path/to/embedding_clips` (same — this directory is the install root).

```bash
cd embedding_clips   # directory with pyproject.toml
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# CPU torch (default PyPI) — or install CUDA wheels from https://pytorch.org first
pip install -U pip
pip install -e .

# optional CLIP backbones
pip install -e ".[clip]"
```

CUDA example (adjust to your driver / CUDA version):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

After install you can use either:

```bash
embedding-clips estimate
python -m embedding_clips estimate
```

### Environment (optional)

```bash
# Linux / macOS
export CRICKET_DATA=/path/to/cricket_train
export EMBEDDING_OUT=/path/to/out

# Windows PowerShell
$env:CRICKET_DATA = "D:\data\cricket_train"
$env:EMBEDDING_OUT = "D:\emb\out"
```

See `.env.example`. You can always pass `--data` / `--videos` / `--out` instead.

## Data layout

`--data` points at a Score-style train root:

```
labels.json
segments/<id>.mp4      # preferred (delivery only)
videos/<stem>.mp4      # fallback
```

Or a flat folder:

```bash
embedding-clips build --videos /path/to/mp4s --limit 50
```

## Quick start

```bash
embedding-clips estimate --backbone dinov2_vits14 --frames 8
embedding-clips estimate --backbone dinov2_vits14 --bench

embedding-clips build --videos ./clips --limit 50 --backbone dinov2_vits14 --device cuda
embedding-clips build --data "$CRICKET_DATA" --backbone dinov2_vits14 --device cuda

embedding-clips query ./clip.mp4 --backbone dinov2_vits14 --topk 10 --show-gt
embedding-clips validate --backbone dinov2_vits14
embedding-clips near-dups --backbone dinov2_vits14 --threshold 0.92
```

Store (default `./out/<backbone>/`):

```
embeddings.npy   # (N, D) float32, L2-normalised
ids.json
meta.json
```

## Multi-GPU / Kaggle T4×2

```bash
embedding-clips build --data "$DATA" --out /kaggle/working/emb \
  --backbone dinov2_vits14 --device cuda:0 --shard-i 0 --shard-n 2

embedding-clips build --data "$DATA" --out /kaggle/working/emb \
  --backbone dinov2_vits14 --device cuda:1 --shard-i 1 --shard-n 2

embedding-clips merge --backbone dinov2_vits14 --out /kaggle/working/emb \
  /kaggle/working/emb/dinov2_vits14_shard0of2 \
  /kaggle/working/emb/dinov2_vits14_shard1of2
```

Or run `bash kaggle_build.sh` after setting `DATA` / `OUT`. Keep only npy/json under `/kaggle/working` (~20 GB limit).

## CLI

| Command | Purpose |
|---|---|
| `estimate` | Time / storage (+ optional `--bench`) |
| `build` | Embed clips (`--limit`, `--shard-i/n`, `--fresh`) |
| `query` | Top-k cosine neighbours (`--url` supported) |
| `merge` | Concatenate shard directories |
| `validate` | Self-retrieval @1 |
| `near-dups` | Pairs with cos ≥ threshold |

## Cost (≈13.8k × 8 frames, videos already on disk)

| Backbone | GPU E2E | Store |
|---|---|---|
| dinov2_vits14 | ~0.5–1.0 h | ~21 MB |
| dinov2_vitb14 | ~0.8–1.5 h | ~42 MB |
| clip_vitb32 | ~0.7–1.2 h | ~28 MB |

## Library usage

```python
from pathlib import Path
from embedding_clips.encoder import device_str, load_encoder, embed_frames
from embedding_clips.store import EmbeddingStore
from embedding_clips.video import sample_frames

device = device_str("cuda")  # or "cpu"
encode, dim, size = load_encoder("dinov2_vits14", device)
frames = sample_frames("clip.mp4", n=8)
vec = embed_frames(frames, encode, size, device)

store = EmbeddingStore(Path("out"), "dinov2_vits14")
hits = store.topk(vec, k=10)
```

## What is not included

- Scene / delivery finder (crop segments before build/query)
- Challenge scoring / GT lookup
- Training labels download

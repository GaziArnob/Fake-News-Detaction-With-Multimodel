"""Features for the MultiBanFakeDetect-style baseline.

MultiBanFakeDetect [3] is the closest prior system to this work in language and
modality scope, and the conference version compared against text-only baselines
instead. This extracts its two feature families so the architecture can be run
under our own protocol:

  DenseNet-169   ImageNet-pretrained, global-average-pooled penultimate layer
                 (1664-d), the visual backbone of that system
  mBERT          bert-base-multilingual-cased, mean-pooled last hidden state
                 (768-d), over the BLIP caption

Both are extracted twice -- on the original files and on the artefact-
controlled re-encodings -- so the comparison is available in both regimes.

Caches to _cache/. Output: nothing but caches; see densenet_baseline.py.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import CACHE, ROOT, reencode

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def densenet_features(paths, name, batch=16):
    """1664-d global-pooled DenseNet-169 features."""
    f = CACHE / f"densenet_{name}.npz"
    cache = {}
    if f.exists():
        b = np.load(f, allow_pickle=True)
        cache = {p: v for p, v in zip(b["paths"].tolist(), b["vectors"])}
    todo = [str(p) for p in paths if str(p) not in cache]
    if not todo:
        print(f"  densenet_{name}: {len(cache)} cached")
        return cache

    from torchvision.models import DenseNet169_Weights, densenet169
    print(f"  densenet_{name}: extracting {len(todo)} on {DEV}", flush=True)
    model = densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1).to(DEV).eval()
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk, tensors, ok = todo[i:i + batch], [], []
        for p in chunk:
            try:
                im = Image.open(p).convert("RGB").resize((224, 224),
                                                         Image.BILINEAR)
            except Exception as exc:
                print(f"    skip {Path(p).name}: {exc}")
                continue
            t = torch.from_numpy(np.asarray(im, np.float32) / 255.0)
            tensors.append(((t.permute(2, 0, 1) - IMAGENET_MEAN)
                            / IMAGENET_STD))
            ok.append(p)
        if not tensors:
            continue
        with torch.inference_mode():
            x = torch.stack(tensors).to(DEV)
            feat = torch.nn.functional.relu(model.features(x))
            feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1)
            feat = torch.flatten(feat, 1).cpu().numpy().astype(np.float32)
        for p, v in zip(ok, feat):
            cache[p] = v
        if (i // batch) % 20 == 0:
            done = i + len(chunk)
            print(f"    {done}/{len(todo)} "
                  f"({done / max(time.time() - t0, 1e-9):.1f}/s)", flush=True)
    np.savez_compressed(f, paths=np.array(list(cache), dtype=object),
                        vectors=np.stack(list(cache.values())))
    print(f"  densenet_{name}: cached {len(cache)}")
    return cache


def mbert_features(captions, name, batch=32):
    """768-d mean-pooled mBERT embeddings of the captions."""
    f = CACHE / f"mbert_{name}.npz"
    if f.exists():
        b = np.load(f, allow_pickle=True)
        print(f"  mbert_{name}: {len(b['vectors'])} cached")
        return b["vectors"]

    from transformers import AutoModel, AutoTokenizer
    print(f"  mbert_{name}: encoding {len(captions)} on {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model = AutoModel.from_pretrained("bert-base-multilingual-cased").to(DEV).eval()
    out = []
    for i in range(0, len(captions), batch):
        chunk = [c if isinstance(c, str) and c.strip() else " "
                 for c in captions[i:i + batch]]
        enc = tok(chunk, padding=True, truncation=True, max_length=128,
                  return_tensors="pt").to(DEV)
        with torch.inference_mode():
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        out.append(pooled.cpu().numpy().astype(np.float32))
        if (i // batch) % 20 == 0:
            print(f"    {i + len(chunk)}/{len(captions)}", flush=True)
    vectors = np.concatenate(out)
    np.savez_compressed(f, vectors=vectors)
    print(f"  mbert_{name}: cached {len(vectors)}")
    return vectors


if __name__ == "__main__":
    feats = pd.read_csv(ROOT / "extracted_features.csv", low_memory=False)
    print(f"training rows: {len(feats)}")

    print("\nRAW images:")
    densenet_features(feats["image_path"].tolist(), "raw")

    print("\nRE-ENCODED images:")
    enc_dir = CACHE / "enc_base"
    enc_paths = []
    for p, lab in zip(feats["image_path"], feats["label"]):
        dst = enc_dir / lab / (Path(p).stem + ".jpg")
        if reencode(Path(p), dst):
            enc_paths.append(str(dst))
    print(f"  re-encoded {len(enc_paths)}")
    densenet_features(enc_paths, "ctl")

    print("\nCAPTIONS (mBERT):")
    mbert_features(feats["caption"].fillna(" ").tolist(), "caption")
    print("\ndone")

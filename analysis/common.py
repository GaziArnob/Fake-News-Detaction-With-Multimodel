"""Shared loaders for the BanFakeMM analysis scripts.

Caches live in _cache/ under the project root, which survives between
sessions. Nothing here recomputes an embedding that is already cached, so the
scripts are cheap to re-run once the cache is warm.

Cache files:
  clip_adapt_raw.npz   300 adaptation images, original files
  clip_adapt_ctl.npz   the same images re-encoded
  clip_big_raw.npz   1,350 untouched sealed candidates, original files
  clip_big_ctl.npz   1,308 survivors, re-encoded
  hash_ref.npz       pHash/dHash for base_train + gen_adapt
  hash_cand.npz      pHash/dHash for the 1,350 candidates
  enc_adapt/, enc_big/   the re-encoded JPEGs themselves
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"E:\Project\Fake News Detection Data")
CACHE = ROOT / "_cache"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dct

CLIP_ID = "openai/clip-vit-base-patch32"
TARGET_SIZE, JPEG_QUALITY = (384, 384), 90
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
TAU = {"raw": {"zeroshot": 0.09, "adapted": 0.12},
       "controlled": {"zeroshot": 0.25, "adapted": 0.35}}
_clip: dict = {}


# --------------------------------------------------------------- embeddings
def load_cache(name):
    f = CACHE / f"clip_{name}.npz"
    if not f.exists():
        return {}
    b = np.load(f, allow_pickle=True)
    return {p: v for p, v in zip(b["paths"].tolist(), b["vectors"])}


def embed(paths, name, batch=32):
    """CLIP image embeddings, identical recipe to news_guard/features.py:
    get_image_features -> pooler_output -> L2 normalise, 512-d float32."""
    import torch
    from transformers import CLIPModel, CLIPProcessor

    cache = load_cache(name)
    todo = [str(p) for p in paths if str(p) not in cache]
    if not todo:
        return cache
    if not _clip:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _clip.update(dev=dev, proc=CLIPProcessor.from_pretrained(CLIP_ID),
                     model=CLIPModel.from_pretrained(CLIP_ID).to(dev).eval())
    print(f"  embedding {len(todo)} for {name}", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), batch):
        chunk, imgs, ok = todo[i:i + batch], [], []
        for p in chunk:
            try:
                imgs.append(Image.open(p).convert("RGB")); ok.append(p)
            except Exception as exc:
                print(f"    skip {Path(p).name}: {exc}", flush=True)
        if not imgs:
            continue
        inp = _clip["proc"](images=imgs, return_tensors="pt").to(_clip["dev"])
        with torch.inference_mode():
            o = _clip["model"].get_image_features(**inp)
            o = getattr(o, "pooler_output", o)
            o = torch.nn.functional.normalize(o, dim=1).cpu().numpy()
        for p, v in zip(ok, o.astype(np.float32)):
            cache[p] = v
        if (i // batch) % 15 == 0:
            print(f"    {i + len(chunk)}/{len(todo)} "
                  f"({(i + len(chunk)) / max(time.time() - t0, 1e-9):.1f}/s)",
                  flush=True)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(CACHE / f"clip_{name}.npz",
                        paths=np.array(list(cache), dtype=object),
                        vectors=np.stack(list(cache.values())))
    return cache


# -------------------------------------------------------------------- hashes
def phash(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((32, 32), Image.LANCZOS), float)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8].flatten()
    return (d > np.median(d[1:])).astype(np.uint8)


def dhash(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten().astype(np.uint8)


def hashes(paths, name):
    f = CACHE / f"hash_{name}.npz"
    if f.exists():
        b = np.load(f, allow_pickle=True)
        return b["paths"].tolist(), b["ph"], b["dh"]
    print(f"  hashing {len(paths)} for {name}", flush=True)
    ok, ph, dh = [], [], []
    for p in paths:
        try:
            ph.append(phash(p)); dh.append(dhash(p)); ok.append(str(p))
        except Exception as exc:
            print(f"    skip {Path(p).name}: {exc}")
    ph, dh = np.array(ph, np.uint8), np.array(dh, np.uint8)
    np.savez_compressed(f, paths=np.array(ok, dtype=object), ph=ph, dh=dh)
    return ok, ph, dh


def min_hamming(cand, ref, chunk=128):
    best = np.full(len(cand), 999, np.int16)
    for i in range(0, len(cand), chunk):
        best[i:i + chunk] = (cand[i:i + chunk, None, :]
                             != ref[None, :, :]).sum(2).min(1)
    return best


def reencode(src, dst):
    """384x384 RGB JPEG q90, no EXIF -- the control of Sec. III-C."""
    if Path(dst).exists():
        return True
    try:
        with Image.open(src) as im:
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            im.convert("RGB").resize(TARGET_SIZE, Image.LANCZOS).save(
                dst, "JPEG", quality=JPEG_QUALITY)
        return True
    except Exception as exc:
        print(f"    re-encode failed {Path(src).name}: {exc}")
        return False


# ------------------------------------------------------------------ features
def base_raw_features():
    """The 2,000 training images, original files."""
    df = pd.read_csv(ROOT / "clip_visual_features.csv", low_memory=False)
    cols = [c for c in df.columns if re.fullmatch(r"clip_\d+", c)]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df, cols


def base_ctl_features():
    """The same 2,000 images, re-encoded."""
    df = pd.read_csv(ROOT / "clip_visual_features_normalized.csv",
                     low_memory=False)
    cols = [c for c in df.columns if re.fullmatch(r"clip_\d+", c)]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df, cols


def clusters():
    return pd.read_csv(ROOT / "duplicate_clusters.csv")


def metrics(prob, is_fake, tau):
    pred = np.asarray(prob) >= tau
    is_fake = np.asarray(is_fake, bool)
    tp = int((pred & is_fake).sum()); fp = int((pred & ~is_fake).sum())
    fn = int((~pred & is_fake).sum()); tn = int((~pred & ~is_fake).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return dict(tau=float(tau), acc=(tp + tn) / len(is_fake) * 100,
                prec_f=p * 100, rec_f=r * 100,
                rec_r=tn / (tn + fp) * 100 if tn + fp else 0.0,
                f1_f=2 * p * r / (p + r) * 100 if p + r else 0.0,
                tp=tp, fp=fp, fn=fn, tn=tn)


def select_tau(prob, is_fake, grid=None):
    """argmax fake-class F1, ties broken by the median of the tied run."""
    grid = np.round(np.arange(0.01, 1.00, 0.01), 2) if grid is None else grid
    sweep = pd.DataFrame([metrics(prob, is_fake, t) for t in grid])
    best = sweep["f1_f"].max()
    tied = sweep.loc[sweep["f1_f"] >= best - 1e-12, "tau"].to_numpy()
    return float(tied[np.abs(tied - np.median(tied)).argmin()]), sweep


SPLIT_IMAGES = {
    "AdversarialEdit": ROOT / "images",
    "AdversarialGen": DATA / "AdversarialGen" / "AdversarialGen" / "images",
    "LinguisticAttack": DATA / "LinguisticAttack" / "LinguisticAttack" / "images",
    "RandomSwap": DATA / "RandomSwap" / "RandomSwap" / "images",
    "SemanticSwap": DATA / "_extracted_semanticswap" / "SemanticSwap" / "images",
}
AUTH_POOL = DATA / "master_enriched" / "master_enriched" / "images" / "real"

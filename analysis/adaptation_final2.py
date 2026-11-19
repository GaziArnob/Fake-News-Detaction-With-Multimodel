"""Multi-seed, threshold-free adaptation contrast -- self-rebuilding.

The scratch caches were wiped, so this rebuilds everything it needs into a
DURABLE cache under the project (_cache/) and then runs the analysis:

  * delta-AUROC with a paired bootstrap CI over the sealed images
    (threshold-free, so it does not depend on where the operating point lands)
  * eight seeds, so the claim does not rest on one XGBoost fit

Sealed set: the 1,350 untouched AdversarialGen images, screened with the full
protocol (pHash<=2 OR dHash<=2 OR CLIP cos>=0.98 against base_train+gen_adapt).
Thresholds fixed in advance from the 148-image validation partition.

Outputs: adaptation_final.csv, adaptation_final_seeds.csv
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(r"E:\Project\Fake News Detection")
DATA = Path(r"E:\Project\Fake News Detection Data")
CACHE = ROOT / "_cache"
CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.fft import dct
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPProcessor

import evaluate_ablation as ab

CLIP_ID = "openai/clip-vit-base-patch32"
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
TAU = {"raw": {"zeroshot": 0.09, "adapted": 0.12},
       "controlled": {"zeroshot": 0.25, "adapted": 0.35}}
_clip = {}


def embed(paths, name):
    f = CACHE / f"clip_{name}.npz"
    cache = {}
    if f.exists():
        b = np.load(f, allow_pickle=True)
        cache = {p: v for p, v in zip(b["paths"].tolist(), b["vectors"])}
    todo = [str(p) for p in paths if str(p) not in cache]
    if todo:
        if not _clip:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _clip["dev"] = dev
            _clip["proc"] = CLIPProcessor.from_pretrained(CLIP_ID)
            _clip["model"] = CLIPModel.from_pretrained(CLIP_ID).to(dev).eval()
        print(f"  embedding {len(todo)} for {name}", flush=True)
        t0 = time.time()
        for i in range(0, len(todo), 32):
            chunk = todo[i:i + 32]
            imgs, ok = [], []
            for p in chunk:
                try:
                    imgs.append(Image.open(p).convert("RGB")); ok.append(p)
                except Exception as e:
                    print(f"    skip {Path(p).name}: {e}", flush=True)
            if not imgs:
                continue
            inp = _clip["proc"](images=imgs, return_tensors="pt").to(_clip["dev"])
            with torch.inference_mode():
                o = _clip["model"].get_image_features(**inp)
                o = getattr(o, "pooler_output", o)
                o = torch.nn.functional.normalize(o, dim=1).cpu().numpy()
            for p, v in zip(ok, o.astype(np.float32)):
                cache[p] = v
            if (i // 32) % 15 == 0:
                print(f"    {i + len(chunk)}/{len(todo)} "
                      f"({(i + len(chunk)) / max(time.time() - t0, 1e-9):.1f}/s)",
                      flush=True)
        np.savez_compressed(f, paths=np.array(list(cache), dtype=object),
                            vectors=np.stack(list(cache.values())))
    return cache


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
        except Exception as e:
            print(f"    skip {Path(p).name}: {e}")
    ph, dh = np.array(ph, np.uint8), np.array(dh, np.uint8)
    np.savez_compressed(f, paths=np.array(ok, dtype=object), ph=ph, dh=dh)
    return ok, ph, dh


def min_h(c, r, chunk=128):
    best = np.full(len(c), 999, np.int16)
    for i in range(0, len(c), chunk):
        best[i:i + chunk] = (c[i:i + chunk, None, :] != r[None, :, :]).sum(2).min(1)
    return best


def reencode(src, dst):
    if dst.exists():
        return True
    try:
        with Image.open(src) as im:
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.convert("RGB").resize((384, 384), Image.LANCZOS).save(
                dst, "JPEG", quality=90)
        return True
    except Exception as e:
        print(f"    re-encode failed {Path(src).name}: {e}")
        return False


print("REBUILDING CACHES (durable, under _cache/)\n", flush=True)
clusters = pd.read_csv(ROOT / "duplicate_clusters.csv")

base_raw = pd.read_csv(ROOT / "clip_visual_features.csv", low_memory=False)
rcols = [c for c in base_raw.columns if re.fullmatch(r"clip_\d+", c)]
for c in rcols:
    base_raw[c] = pd.to_numeric(base_raw[c], errors="coerce").fillna(0.0)
seen = {p: v for p, v in zip(base_raw["image_path"],
                             base_raw[rcols].to_numpy(np.float32))}

adapt_rows = clusters[clusters.partition == "gen_adapt"].reset_index(drop=True)
seen.update(embed(adapt_rows["image_path"].tolist(), "adapt_raw"))

used = set(clusters["image_path"])
gen = DATA / "AdversarialGen" / "AdversarialGen" / "images"
fresh = [str(p) for sub in ("real", "fake") for p in (gen / sub).iterdir()
         if p.is_file() and str(p) not in used]
print(f"untouched images: {len(fresh)}", flush=True)
raw = embed(fresh, "big_raw")

ref_paths = clusters.loc[clusters.partition.isin(["base_train", "gen_adapt"]),
                         "image_path"].tolist()
ref_ok, ref_ph, ref_dh = hashes(ref_paths, "ref")
cand_ok, cand_ph, cand_dh = hashes(fresh, "cand")

ref_vec = np.stack([seen[p] for p in ref_paths if p in seen])
ref_vec /= np.linalg.norm(ref_vec, axis=1, keepdims=True)
drop = (min_h(cand_ph, ref_ph) <= 2) | (min_h(cand_dh, ref_dh) <= 2)
drop |= np.array([float((ref_vec @ (raw[p] / np.linalg.norm(raw[p]))).max())
                  if p in raw else 0.0 for p in cand_ok]) >= 0.98
keep = [p for p, d in zip(cand_ok, drop) if not d]
is_fake = np.array([Path(p).parent.name == "fake" for p in keep])
truth = is_fake.astype(int)
print(f"screened sealed set: {len(keep)} ({int(truth.sum())} manipulated)\n",
      flush=True)

ADAPT_ENC, BIG_ENC = CACHE / "enc_adapt", CACHE / "enc_big"
a_enc = [str(ADAPT_ENC / (Path(p).stem + ".jpg")) for p in adapt_rows["image_path"]
         if reencode(Path(p), ADAPT_ENC / (Path(p).stem + ".jpg"))]
b_enc = [str(BIG_ENC / (Path(p).stem + ".jpg")) for p in keep
         if reencode(Path(p), BIG_ENC / (Path(p).stem + ".jpg"))]
ctl_adapt = embed(a_enc, "adapt_ctl")
ctl_big = embed(b_enc, "big_ctl")

norm = pd.read_csv(ROOT / "clip_visual_features_normalized.csv", low_memory=False)
ncols = [c for c in norm.columns if re.fullmatch(r"clip_\d+", c)]
for c in ncols:
    norm[c] = pd.to_numeric(norm[c], errors="coerce").fillna(0.0)


def data(mode):
    if mode == "raw":
        X = pd.DataFrame([seen[p] for p in base_raw["image_path"]], columns=rcols)
        y = base_raw["label"].eq("fake").astype(int).to_numpy()
        a = adapt_rows[adapt_rows["image_path"].isin(seen)]
        aX = pd.DataFrame([seen[p] for p in a["image_path"]], columns=rcols)
        aY = a["label"].eq("fake").astype(int).to_numpy()
        Xt = pd.DataFrame([raw[p] for p in keep], columns=rcols)
        return rcols, X, y, aX, aY, Xt
    stem_lab = dict(zip(adapt_rows["image_path"].map(lambda s: Path(s).stem),
                        adapt_rows["label"]))
    ap = [p for p in a_enc if p in ctl_adapt]
    aX = pd.DataFrame([ctl_adapt[p] for p in ap], columns=ncols)
    aY = np.array([stem_lab.get(Path(p).stem) == "fake" for p in ap]).astype(int)
    Xt = pd.DataFrame([ctl_big[str(BIG_ENC / (Path(p).stem + ".jpg"))]
                       for p in keep], columns=ncols)
    return (ncols, norm[ncols].reset_index(drop=True),
            norm["label"].eq("fake").astype(int).to_numpy(), aX, aY, Xt)


rng = np.random.default_rng(42)
seed_rows, summary = [], []
for mode in ("raw", "controlled"):
    cols, X, y, aX, aY, Xt = data(mode)
    print(f"{mode.upper()}: train {len(X)}, adapt {len(aX)} "
          f"({int(aY.sum())} manipulated), sealed {len(Xt)}", flush=True)
    probs = {"zeroshot": [], "adapted": []}
    for s in SEEDS:
        ab.SEED = s
        row = {"mode": mode, "seed": s}
        for name, (TX, TY) in {
                "zeroshot": (X, y),
                "adapted": (pd.concat([X, aX], ignore_index=True),
                            np.concatenate([y, aY]))}.items():
            m = ab.build_pipeline(False, [], True, cols)
            m.fit(TX, TY)
            p = m.predict_proba(Xt)[:, 1]
            probs[name].append(p)
            row[f"auroc_{name}"] = roc_auc_score(truth, p)
            row[f"acc_{name}"] = float(((p >= TAU[mode][name]) == is_fake).mean() * 100)
        row["d_auroc"] = row["auroc_adapted"] - row["auroc_zeroshot"]
        row["d_acc"] = row["acc_adapted"] - row["acc_zeroshot"]
        seed_rows.append(row)
        print(f"  seed {s:>3}  dAUROC {row['d_auroc']:+.4f}  "
              f"dAcc {row['d_acc']:+5.1f}", flush=True)

    d_auc = np.array([r["d_auroc"] for r in seed_rows if r["mode"] == mode])
    d_acc = np.array([r["d_acc"] for r in seed_rows if r["mode"] == mode])
    pz, pa = np.mean(probs["zeroshot"], 0), np.mean(probs["adapted"], 0)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, len(truth), len(truth))
        if truth[idx].min() == truth[idx].max():
            continue
        boot.append(roc_auc_score(truth[idx], pa[idx])
                    - roc_auc_score(truth[idx], pz[idx]))
    boot = np.array(boot)
    point = roc_auc_score(truth, pa) - roc_auc_score(truth, pz)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    summary.append({"mode": mode, "n": len(truth), "seeds": len(SEEDS),
                    "auroc_zeroshot": round(float(roc_auc_score(truth, pz)), 4),
                    "auroc_adapted": round(float(roc_auc_score(truth, pa)), 4),
                    "d_auroc": round(float(point), 4),
                    "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                    "d_auroc_seed_sd": round(float(d_auc.std()), 4),
                    "d_acc_mean": round(float(d_acc.mean()), 2),
                    "d_acc_seed_sd": round(float(d_acc.std()), 2)})
    print(f"  => dAUROC {point:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"seed sd {d_auc.std():.4f}")
    print(f"     dAcc {d_acc.mean():+.1f} +- {d_acc.std():.1f} across seeds\n",
          flush=True)

pd.DataFrame(seed_rows).to_csv(ROOT / "adaptation_final_seeds.csv", index=False)
pd.DataFrame(summary).to_csv(ROOT / "adaptation_final.csv", index=False)
print("wrote adaptation_final.csv and adaptation_final_seeds.csv")

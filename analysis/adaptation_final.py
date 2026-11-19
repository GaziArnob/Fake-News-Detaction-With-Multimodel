"""Journal-grade version of the adaptation contrast.

Adds the two things the result was missing:
  1. a paired bootstrap confidence interval on delta-AUROC, which is now the
     headline statistic (threshold-free, so it does not depend on where the
     operating point happens to land);
  2. eight seeds, so the claim does not rest on one lucky XGBoost fit.

Sealed set: the 1,308 untouched AdversarialGen images, screened with the full
protocol (pHash<=2 OR dHash<=2 OR CLIP cos>=0.98 against base_train+gen_adapt).
Thresholds stay fixed in advance from the 148-image validation partition.

Outputs: adaptation_final.csv, adaptation_final_seeds.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(r"E:\Project\Fake News Detection")
HERE = Path(__file__).parent
SCRATCH = HERE / "scratchpad"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import evaluate_ablation as ab

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
TAU = {"raw": {"zeroshot": 0.09, "adapted": 0.12},
       "controlled": {"zeroshot": 0.25, "adapted": 0.35}}


def min_h(c, r, chunk=128):
    best = np.full(len(c), 999, dtype=np.int16)
    for i in range(0, len(c), chunk):
        best[i:i + chunk] = (c[i:i + chunk, None, :] != r[None, :, :]).sum(2).min(1)
    return best


clusters = pd.read_csv(ROOT / "duplicate_clusters.csv")

base_raw = pd.read_csv(ROOT / "clip_visual_features.csv", low_memory=False)
rcols = [c for c in base_raw.columns if re.fullmatch(r"clip_\d+", c)]
for c in rcols:
    base_raw[c] = pd.to_numeric(base_raw[c], errors="coerce").fillna(0.0)
seen = {p: v for p, v in zip(base_raw["image_path"],
                             base_raw[rcols].to_numpy(dtype=np.float32))}
ab_blob = np.load(SCRATCH / "clip_adapt_raw.npz", allow_pickle=True)
seen.update({p: v for p, v in zip(ab_blob["paths"].tolist(), ab_blob["vectors"])})

raw_blob = np.load(SCRATCH / "clip_big_raw.npz", allow_pickle=True)
raw = {p: v for p, v in zip(raw_blob["paths"].tolist(), raw_blob["vectors"])}
ctl_blob = np.load(SCRATCH / "clip_big_ctl.npz", allow_pickle=True)
ctl = {p: v for p, v in zip(ctl_blob["paths"].tolist(), ctl_blob["vectors"])}

# rebuild the screened sealed set
hb = np.load(SCRATCH / "hashes_cand.npz", allow_pickle=True)
hr = np.load(SCRATCH / "hashes_ref.npz", allow_pickle=True)
ref_paths = clusters.loc[clusters.partition.isin(["base_train", "gen_adapt"]),
                         "image_path"].tolist()
ref_vec = np.stack([seen[p] for p in ref_paths if p in seen])
ref_vec /= np.linalg.norm(ref_vec, axis=1, keepdims=True)
cand = hb["paths"].tolist()
drop = (min_h(hb["ph"], hr["ph"]) <= 2) | (min_h(hb["dh"], hr["dh"]) <= 2)
drop |= np.array([float((ref_vec @ (raw[p] / np.linalg.norm(raw[p]))).max())
                  if p in raw else 0.0 for p in cand]) >= 0.98
keep = [p for p, d in zip(cand, drop) if not d]
is_fake = np.array(["fake" in Path(p).parent.name for p in keep])
print(f"sealed set {len(keep)} ({int(is_fake.sum())} manipulated)", flush=True)

norm = pd.read_csv(ROOT / "clip_visual_features_normalized.csv", low_memory=False)
ncols = [c for c in norm.columns if re.fullmatch(r"clip_\d+", c)]
for c in ncols:
    norm[c] = pd.to_numeric(norm[c], errors="coerce").fillna(0.0)
gb = np.load(SCRATCH / "clip_gen_reencoded.npz", allow_pickle=True)
gc = {p: v for p, v in zip(gb["paths"].tolist(), gb["vectors"])}
lab = dict(zip(clusters["image_path"].map(lambda s: Path(s).stem),
               clusters["label"]))
ap = [p for p in gc if Path(p).parent.name == "adapt"]


def data(mode):
    if mode == "raw":
        X = pd.DataFrame([seen[p] for p in base_raw["image_path"]], columns=rcols)
        y = base_raw["label"].eq("fake").astype(int).to_numpy()
        a = clusters[(clusters.partition == "gen_adapt")
                     & clusters["image_path"].isin(seen)]
        aX = pd.DataFrame([seen[p] for p in a["image_path"]], columns=rcols)
        aY = a["label"].eq("fake").astype(int).to_numpy()
        Xt = pd.DataFrame([raw[p] for p in keep], columns=rcols)
        return rcols, X, y, aX, aY, Xt
    X = norm[ncols].reset_index(drop=True)
    y = norm["label"].eq("fake").astype(int).to_numpy()
    aX = pd.DataFrame([gc[p] for p in ap], columns=ncols)
    aY = np.array([lab.get(Path(p).stem) == "fake" for p in ap]).astype(int)
    Xt = pd.DataFrame([ctl[str(SCRATCH / "big_reencoded" / (Path(p).stem + ".jpg"))]
                       for p in keep], columns=ncols)
    return ncols, X, y, aX, aY, Xt


rng = np.random.default_rng(42)
seed_rows, summary = [], []
truth = is_fake.astype(int)

for mode in ("raw", "controlled"):
    cols, X, y, aX, aY, Xt = data(mode)
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
        print(f"  {mode:<11} seed {s:>3}  dAUROC {row['d_auroc']:+.4f}  "
              f"dAcc {row['d_acc']:+5.1f}", flush=True)

    d_auc = np.array([r["d_auroc"] for r in seed_rows if r["mode"] == mode])
    d_acc = np.array([r["d_acc"] for r in seed_rows if r["mode"] == mode])

    # paired bootstrap over IMAGES, on seed-averaged probabilities
    pz = np.mean(probs["zeroshot"], axis=0)
    pa = np.mean(probs["adapted"], axis=0)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, len(truth), len(truth))
        if truth[idx].min() == truth[idx].max():
            continue
        boot.append(roc_auc_score(truth[idx], pa[idx])
                    - roc_auc_score(truth[idx], pz[idx]))
    boot = np.array(boot)
    point = roc_auc_score(truth, pa) - roc_auc_score(truth, pz)
    summary.append({"mode": mode, "n": len(truth), "seeds": len(SEEDS),
                    "d_auroc": round(float(point), 4),
                    "ci_lo": round(float(np.percentile(boot, 2.5)), 4),
                    "ci_hi": round(float(np.percentile(boot, 97.5)), 4),
                    "d_auroc_seed_sd": round(float(d_auc.std()), 4),
                    "d_acc_mean": round(float(d_acc.mean()), 2),
                    "d_acc_seed_sd": round(float(d_acc.std()), 2)})
    print(f"\n  {mode.upper()}  dAUROC {point:+.4f} "
          f"95% CI [{np.percentile(boot, 2.5):+.4f}, "
          f"{np.percentile(boot, 97.5):+.4f}]   "
          f"seed sd {d_auc.std():.4f}")
    print(f"  {'':<11} dAcc {d_acc.mean():+.1f} +- {d_acc.std():.1f} "
          f"across seeds\n", flush=True)

pd.DataFrame(seed_rows).to_csv(ROOT / "adaptation_final_seeds.csv", index=False)
pd.DataFrame(summary).to_csv(ROOT / "adaptation_final.csv", index=False)
print("wrote adaptation_final.csv and adaptation_final_seeds.csv")

"""MultiBanFakeDetect-style baseline under our own protocol.

The conference version compared only against text-only baselines, which is the
objection a reviewer would raise first: Section II names MultiBanFakeDetect as
the closest prior system and never compares against it.

This runs that system's feature families through the identical evaluation
pipeline -- StratifiedGroupKFold over the duplicate clusters, 2 repeats x
5 folds, same PCA reduction, same XGBoost, same tau = 0.50 cut-off -- so the
only thing that differs is the backbone:

  DenseNet-169                   their visual backbone, vs our CLIP
  DenseNet-169 + mBERT caption   their hybrid fusion, vs our CLIP + CISF

Both regimes are reported: original files, and artefact-controlled
re-encodings, since Section IV-C showed the former carries a shortcut.

Note on what this is and is not. It is a comparison of feature families and
fusion structure under one protocol. It is not a reimplementation of their
trained system: MultiBanFakeDetect consumes the article's own text, which our
pipeline never reads, so its mBERT branch here encodes the BLIP caption
instead. The comparison therefore isolates the backbone, and we say so.

Output: densenet_baseline.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from common import CACHE, ROOT, base_ctl_features, base_raw_features, clusters

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SEED, N_SPLITS, N_REPEATS = 42, 5, 2

# Identical to scripts/evaluate_ablation.py's classifier.
XGB = dict(max_depth=3, learning_rate=0.0742, n_estimators=443,
           subsample=0.8661185283697008, colsample_bytree=0.683,
           min_child_weight=2, gamma=0.143, reg_alpha=1.01e-5,
           reg_lambda=4.911, random_state=SEED, n_jobs=4,
           tree_method="hist", eval_metric="logloss")


def load(name):
    f = CACHE / f"{name}.npz"
    if not f.exists():
        raise SystemExit(f"missing cache {f} -- run extract_densenet_mbert.py")
    b = np.load(f, allow_pickle=True)
    return ({p: v for p, v in zip(b["paths"].tolist(), b["vectors"])}
            if "paths" in b else b["vectors"])


def evaluate(name, X, y, groups, n_pca):
    folds = []
    for r in range(N_REPEATS):
        folds.extend(StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True,
            random_state=SEED + r).split(X, y, groups))
    acc, prec, rec, f1 = [], [], [], []
    for tr, te in folds:
        model = Pipeline([("pca", PCA(n_components=n_pca, random_state=SEED)),
                          ("xgb", XGBClassifier(**XGB))])
        model.fit(X[tr], y[tr])
        pred = (model.predict_proba(X[te])[:, 1] >= 0.5).astype(int)
        t = y[te]
        tp = int(((pred == 1) & (t == 1)).sum())
        fp = int(((pred == 1) & (t == 0)).sum())
        fn = int(((pred == 0) & (t == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r_ = tp / (tp + fn) if tp + fn else 0.0
        acc.append((pred == t).mean() * 100)
        prec.append(p * 100)
        rec.append(r_ * 100)
        f1.append(2 * p * r_ / (p + r_) * 100 if p + r_ else 0.0)
    row = {"configuration": name, "acc": round(float(np.mean(acc)), 2),
           "std": round(float(np.std(acc)), 2),
           "prec_f": round(float(np.mean(prec)), 2),
           "rec_f": round(float(np.mean(rec)), 2),
           "f1_f": round(float(np.mean(f1)), 2)}
    print(f"  {name:<46} {row['acc']:6.2f} +- {row['std']:4.2f}"
          f"  P {row['prec_f']:6.2f}  R {row['rec_f']:6.2f}"
          f"  F1 {row['f1_f']:6.2f}", flush=True)
    return row


feats = pd.read_csv(ROOT / "extracted_features.csv", low_memory=False)
cl = clusters()
cluster_of = dict(zip(cl["image_path"], cl["cluster"]))
feats["cluster"] = feats["image_path"].map(cluster_of)
feats = feats[feats["cluster"].notna()].reset_index(drop=True)
y = feats["label"].eq("fake").astype(int).to_numpy()
groups = feats["cluster"].astype(int).to_numpy()
print(f"rows {len(feats)}  groups {len(np.unique(groups))}  "
      f"fake {int(y.sum())}\n")

dn_raw = load("densenet_raw")
dn_ctl = load("densenet_ctl")
mbert = load("mbert_caption")

X_dn_raw = np.stack([dn_raw[p] for p in feats["image_path"]])
enc = CACHE / "enc_base"
X_dn_ctl = np.stack([dn_ctl[str(enc / lab / (Path(p).stem + ".jpg"))]
                     for p, lab in zip(feats["image_path"], feats["label"])])
X_mbert = mbert[:len(feats)]
print(f"DenseNet {X_dn_raw.shape}   mBERT {X_mbert.shape}\n")

rows = []
print("MULTIBANFAKEDETECT-STYLE BASELINE, our grouped protocol:")
rows.append(evaluate("DenseNet-169 (raw images)", X_dn_raw, y, groups, 64))
rows.append(evaluate("DenseNet-169 (artefact-controlled)", X_dn_ctl, y,
                     groups, 64))

# Their hybrid: visual backbone + mBERT text branch, concatenated.
pca_t = PCA(n_components=32, random_state=SEED).fit_transform(X_mbert)
rows.append(evaluate("DenseNet-169 + mBERT (raw)",
                     np.hstack([X_dn_raw, pca_t]), y, groups, 64))
rows.append(evaluate("DenseNet-169 + mBERT (controlled)",
                     np.hstack([X_dn_ctl, pca_t]), y, groups, 64))

pd.DataFrame(rows).to_csv(ROOT / "densenet_baseline.csv", index=False)
print(f"\nwrote {ROOT / 'densenet_baseline.csv'}")
print("\nreference, same protocol:")
print("  CLIP only (raw)                 95.38 +- 1.33  P 94.68 R 96.15 F1 95.41")
print("  CLIP only (controlled)          94.35 +- 0.94  P 94.17 R 94.60 F1 94.37")
print("  Caption + CISF + CLIP (raw)     95.40 +- 1.09  P 95.13 R 95.70 F1 95.41")

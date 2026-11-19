"""Dataset-artefact control, under the SAME grouped protocol as Table I.

Two configurations, both with StratifiedGroupKFold over the duplicate clusters,
2 repeats x 5 folds, identical hyperparameters, so every number lands in the
same column as the ablation:

  metadata only          width, height, area, aspect, file size, format,
                         colour mode -- no pixel content at all
  CLIP only, re-encoded  384x384 RGB JPEG q90, EXIF dropped

Compare against Table I's CLIP-only on the original files: 95.38 +- 1.33.

Produces the two artefact rows of Table I. Output: artefact_control.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from common import ROOT, base_ctl_features, clusters

sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_ablation as ab

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SEED, N_SPLITS, N_REPEATS = 42, 5, 2


def evaluate(name, X, y, groups, clip_columns=None, numeric=None):
    folds = []
    for r in range(N_REPEATS):
        folds.extend(StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True,
            random_state=SEED + r).split(X, y, groups))

    acc, prec, rec, f1 = [], [], [], []
    for tr, te in folds:
        model = ab.build_pipeline(False, numeric or [], bool(clip_columns),
                                  clip_columns or [])
        model.fit(X.iloc[tr], y[tr])
        pred = (model.predict_proba(X.iloc[te])[:, 1] >= 0.5).astype(int)
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
    print(f"  {row['configuration']:<44} {row['acc']:6.2f} +- {row['std']:4.2f}"
          f"  P {row['prec_f']:6.2f}  R {row['rec_f']:6.2f}"
          f"  F1 {row['f1_f']:6.2f}", flush=True)
    return row


cl = clusters()
cluster_of = dict(zip(cl["image_path"], cl["cluster"]))
norm, cols = base_ctl_features()
key = "image_path" if "image_path" in norm.columns else "source"
norm = norm.copy()
norm["cluster"] = norm[key].map(cluster_of)
norm = norm[norm["cluster"].notna()].reset_index(drop=True)

y = norm["label"].eq("fake").astype(int).to_numpy()
groups = norm["cluster"].astype(int).to_numpy()
print(f"rows {len(norm)}  groups {len(np.unique(groups))}  "
      f"fake {int(y.sum())}\n")

meta = []
for p in norm[key]:
    try:
        with Image.open(p) as im:
            w, h = im.size
            fmt, mode = im.format or "", im.mode
        meta.append({"w": w, "h": h, "area": w * h, "aspect": w / max(h, 1),
                     "size_bytes": Path(p).stat().st_size,
                     "is_png": int(fmt == "PNG"), "is_rgb": int(mode == "RGB")})
    except Exception:
        meta.append(dict.fromkeys(
            ["w", "h", "area", "aspect", "size_bytes", "is_png", "is_rgb"], 0))
meta = pd.DataFrame(meta)

print("GROUPED PROTOCOL, FULL METRIC SET:")
rows = [evaluate("Metadata only (no pixels)", meta, y, groups,
                 numeric=list(meta.columns)),
        evaluate("CLIP only, artefact-controlled images",
                 norm[cols], y, groups, clip_columns=cols)]
pd.DataFrame(rows).to_csv(ROOT / "artefact_control.csv", index=False)
print(f"\nwrote {ROOT / 'artefact_control.csv'}")
print("reference (Table I): CLIP only, raw  95.38 +- 1.33  "
      "P 94.68  R 96.15  F1 95.41")

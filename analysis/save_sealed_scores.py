"""Save per-image scores for the 1,308-image sealed partition.

The sealed-partition results were reported as aggregates only; the per-image
probabilities were never written out, so plots such as an ROC curve had to fall
back on the older 194-image partition. This rebuilds the screened set exactly as
article_screen.py and adaptation_final2.py do (pHash <= 2 OR dHash <= 2 OR CLIP
cosine >= 0.98 against base_train + gen_adapt), fits the four CLIP-only models
with seed 42, and writes every score.

  raw / controlled   original files / re-encoded 384x384 RGB JPEG q90
  zeroshot / adapted trained on AdversarialEdit only / plus 300 gen_adapt images

Check: the seed-42 AUROCs must equal the seed-42 row of adaptation_final_seeds.csv,
since that run used the same pipeline and seed. If they differ, the rebuilt set or
model is not the one the paper reports and nothing here should be used.

Output: sealed1308_scores.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import CACHE, ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_ablation as ab

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(name):
    b = np.load(CACHE / f"clip_{name}.npz", allow_pickle=True)
    return {p: v for p, v in zip(b["paths"].tolist(), b["vectors"])}


def min_h(c, r, chunk=128):
    best = np.full(len(c), 999, np.int16)
    for i in range(0, len(c), chunk):
        best[i:i + chunk] = (c[i:i + chunk, None, :]
                             != r[None, :, :]).sum(2).min(1)
    return best


clusters = pd.read_csv(ROOT / "duplicate_clusters.csv")
base_raw = pd.read_csv(ROOT / "clip_visual_features.csv", low_memory=False)
rcols = [c for c in base_raw.columns if re.fullmatch(r"clip_\d+", c)]
for c in rcols:
    base_raw[c] = pd.to_numeric(base_raw[c], errors="coerce").fillna(0.0)
seen = {p: v for p, v in zip(base_raw["image_path"],
                             base_raw[rcols].to_numpy(np.float32))}
seen.update(load("adapt_raw"))
raw, ctl_big, ctl_adapt = load("big_raw"), load("big_ctl"), load("adapt_ctl")

hr = np.load(CACHE / "hash_ref.npz", allow_pickle=True)
hc = np.load(CACHE / "hash_cand.npz", allow_pickle=True)
ref_paths = clusters.loc[clusters.partition.isin(["base_train", "gen_adapt"]),
                         "image_path"].tolist()
ref_vec = np.stack([seen[p] for p in ref_paths if p in seen])
ref_vec /= np.linalg.norm(ref_vec, axis=1, keepdims=True)
cand = hc["paths"].tolist()
drop = (min_h(hc["ph"], hr["ph"]) <= 2) | (min_h(hc["dh"], hr["dh"]) <= 2)
drop |= np.array([float((ref_vec @ (raw[p] / np.linalg.norm(raw[p]))).max())
                  if p in raw else 0.0 for p in cand]) >= 0.98
keep = [p for p, d in zip(cand, drop) if not d]
label = np.array(["fake" if Path(p).parent.name == "fake" else "real"
                  for p in keep])
truth = (label == "fake").astype(int)
print(f"sealed set: {len(keep)} images, {int(truth.sum())} manipulated")
assert len(keep) == 1308 and int(truth.sum()) == 658, "set differs from the paper"

norm = pd.read_csv(ROOT / "clip_visual_features_normalized.csv", low_memory=False)
ncols = [c for c in norm.columns if re.fullmatch(r"clip_\d+", c)]
for c in ncols:
    norm[c] = pd.to_numeric(norm[c], errors="coerce").fillna(0.0)
adapt_rows = clusters[clusters.partition == "gen_adapt"].reset_index(drop=True)
stem_lab = dict(zip(adapt_rows["image_path"].map(lambda s: Path(s).stem),
                    adapt_rows["label"]))


def data(mode):
    if mode == "raw":
        X = pd.DataFrame([seen[p] for p in base_raw["image_path"]], columns=rcols)
        y = base_raw["label"].eq("fake").astype(int).to_numpy()
        a = adapt_rows[adapt_rows["image_path"].isin(seen)]
        aX = pd.DataFrame([seen[p] for p in a["image_path"]], columns=rcols)
        aY = a["label"].eq("fake").astype(int).to_numpy()
        Xt = pd.DataFrame([raw[p] for p in keep], columns=rcols)
        return rcols, X, y, aX, aY, Xt
    # Row ORDER matters: XGBoost's row subsampling depends on it, so the
    # adaptation images must be in the cluster-file order adaptation_final2.py
    # used, not sorted order.
    ap = [str(CACHE / "enc_adapt" / (Path(p).stem + ".jpg"))
          for p in adapt_rows["image_path"]]
    ap = [p for p in ap if p in ctl_adapt]
    aX = pd.DataFrame([ctl_adapt[p] for p in ap], columns=ncols)
    aY = np.array([stem_lab.get(Path(p).stem) == "fake" for p in ap]).astype(int)
    Xt = pd.DataFrame([ctl_big[str(CACHE / "enc_big" / (Path(p).stem + ".jpg"))]
                       for p in keep], columns=ncols)
    return (ncols, norm[ncols].reset_index(drop=True),
            norm["label"].eq("fake").astype(int).to_numpy(), aX, aY, Xt)


out = pd.DataFrame({"image_path": keep, "label": label})
ab.SEED = 42
for mode in ("raw", "controlled"):
    cols, X, y, aX, aY, Xt = data(mode)
    for name, (TX, TY) in {"zeroshot": (X, y),
                           "adapted": (pd.concat([X, aX], ignore_index=True),
                                       np.concatenate([y, aY]))}.items():
        m = ab.build_pipeline(False, [], True, cols)
        m.fit(TX, TY)
        out[f"prob_{mode}_{name}"] = m.predict_proba(Xt)[:, 1]

out.to_csv(ROOT / "sealed1308_scores.csv", index=False)
print(f"wrote {ROOT / 'sealed1308_scores.csv'}\n")

# ---- the check -------------------------------------------------------------
ref = pd.read_csv(ROOT / "adaptation_final_seeds.csv")
ref = ref[ref.seed == 42].set_index("mode")
print(f"{'model':<24}{'AUROC here':>12}{'seed-42 row of saved CSV':>28}")
worst = 0.0
for mode in ("raw", "controlled"):
    for name in ("zeroshot", "adapted"):
        a = roc_auc_score(truth, out[f"prob_{mode}_{name}"])
        b = float(ref.loc[mode, f"auroc_{name}"])
        worst = max(worst, abs(a - b))
        print(f"{mode + ' ' + name:<24}{a:>12.6f}{b:>28.6f}")
print(f"\nlargest difference: {worst:.2e}")
print("MATCH" if worst < 1e-6 else "MISMATCH: do not use these scores")

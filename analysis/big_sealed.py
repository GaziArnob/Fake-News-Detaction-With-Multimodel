"""The enlarged sealed partition, screened with the FULL protocol, scored
both ways.

The 1,350 AdversarialGen images the study never touched -- not training, not
adaptation, not threshold selection -- are a genuinely pristine test set. This
screens them under all three criteria of Sec. III-D (pHash<=2 OR dHash<=2 OR
CLIP cosine>=0.98 against base_train + gen_adapt) and scores the survivors:

  raw          the original files, carrying the resolution artefact
  controlled   re-encoded 384x384 RGB JPEG q90, artefact removed

Thresholds are fixed in advance from the 148-image validation partition and
are NOT reselected here.

Produces Table III. Output: big_sealed_results_v2.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common import (CACHE, ROOT, SPLIT_IMAGES, TAU, base_ctl_features,
                    base_raw_features, clusters, embed, hashes, min_hamming,
                    reencode)

sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_ablation as ab

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cl = clusters()
used = set(cl["image_path"])
gen = SPLIT_IMAGES["AdversarialGen"]
fresh = [(str(p), sub == "fake") for sub in ("real", "fake")
         for p in (gen / sub).iterdir() if p.is_file() and str(p) not in used]
print(f"untouched images: {len(fresh)} "
      f"({sum(f for _, f in fresh)} manipulated)")

base_raw, rcols = base_raw_features()
seen = {p: v for p, v in zip(base_raw["image_path"],
                             base_raw[rcols].to_numpy(np.float32))}
adapt_rows = cl[cl.partition == "gen_adapt"].reset_index(drop=True)
seen.update(embed(adapt_rows["image_path"].tolist(), "adapt_raw"))
raw = embed([p for p, _ in fresh], "big_raw")

ref_paths = cl.loc[cl.partition.isin(["base_train", "gen_adapt"]),
                   "image_path"].tolist()
ref_ok, ref_ph, ref_dh = hashes(ref_paths, "ref")
cand_ok, cand_ph, cand_dh = hashes([p for p, _ in fresh], "cand")

ref_vec = np.stack([seen[p] for p in ref_paths if p in seen])
ref_vec /= np.linalg.norm(ref_vec, axis=1, keepdims=True)
label_of = dict(fresh)
cos = np.array([float((ref_vec @ (raw[p] / np.linalg.norm(raw[p]))).max())
                if p in raw else 0.0 for p in cand_ok])
drop_ph, drop_dh, drop_cos = (min_hamming(cand_ph, ref_ph) <= 2,
                              min_hamming(cand_dh, ref_dh) <= 2, cos >= 0.98)
drop = drop_ph | drop_dh | drop_cos
print(f"\nscreen on {len(cand_ok)} candidates:")
print(f"  pHash <= 2      : {int(drop_ph.sum())}")
print(f"  dHash <= 2      : {int(drop_dh.sum())}")
print(f"  CLIP cos >= 0.98: {int(drop_cos.sum())}")
print(f"  dropped (union) : {int(drop.sum())}")
keep = [(p, label_of[p]) for p, d in zip(cand_ok, drop) if not d]
print(f"  kept            : {len(keep)} "
      f"({sum(f for _, f in keep)} manipulated)")

BIG_ENC = CACHE / "enc_big"
enc = [(str(BIG_ENC / (Path(p).stem + ".jpg")), f) for p, f in keep
       if reencode(Path(p), BIG_ENC / (Path(p).stem + ".jpg"))]
ctl = embed([p for p, _ in enc], "big_ctl")

ADAPT_ENC = CACHE / "enc_adapt"
a_enc = [str(ADAPT_ENC / (Path(p).stem + ".jpg"))
         for p in adapt_rows["image_path"]
         if reencode(Path(p), ADAPT_ENC / (Path(p).stem + ".jpg"))]
ctl_adapt = embed(a_enc, "adapt_ctl")
norm, ncols = base_ctl_features()
stem_lab = dict(zip(adapt_rows["image_path"].map(lambda s: Path(s).stem),
                    adapt_rows["label"]))


def data(mode):
    if mode == "raw":
        a = adapt_rows[adapt_rows["image_path"].isin(seen)]
        return (rcols,
                pd.DataFrame([seen[p] for p in base_raw["image_path"]],
                             columns=rcols),
                base_raw["label"].eq("fake").astype(int).to_numpy(),
                pd.DataFrame([seen[p] for p in a["image_path"]], columns=rcols),
                a["label"].eq("fake").astype(int).to_numpy(),
                pd.DataFrame([raw[p] for p, _ in keep], columns=rcols),
                np.array([f for _, f in keep]))
    ap = [p for p in a_enc if p in ctl_adapt]
    return (ncols, norm[ncols].reset_index(drop=True),
            norm["label"].eq("fake").astype(int).to_numpy(),
            pd.DataFrame([ctl_adapt[p] for p in ap], columns=ncols),
            np.array([stem_lab.get(Path(p).stem) == "fake"
                      for p in ap]).astype(int),
            pd.DataFrame([ctl[p] for p, _ in enc], columns=ncols),
            np.array([f for _, f in enc]))


rng = np.random.default_rng(42)
rows = []
for mode in ("raw", "controlled"):
    cols, X, y, aX, aY, Xt, is_fake = data(mode)
    print(f"\n{mode.upper()}: sealed test {len(Xt)} "
          f"({int(is_fake.sum())} manipulated)")
    correct = {}
    for name, (TX, TY) in {"zeroshot": (X, y),
                           "adapted": (pd.concat([X, aX], ignore_index=True),
                                       np.concatenate([y, aY]))}.items():
        m = ab.build_pipeline(False, [], True, cols)
        m.fit(TX, TY)
        pred = m.predict_proba(Xt)[:, 1] >= TAU[mode][name]
        correct[name] = (pred == is_fake).astype(int)
        tp = int((pred & is_fake).sum()); fp = int((pred & ~is_fake).sum())
        fn = int((~pred & is_fake).sum()); tn = int((~pred & ~is_fake).sum())
        boot = [correct[name][rng.integers(0, len(Xt), len(Xt))].mean() * 100
                for _ in range(5000)]
        row = dict(mode=mode, model=name, tau=TAU[mode][name], n=len(Xt),
                   acc=round(correct[name].mean() * 100, 1),
                   prec_f=round(tp / max(tp + fp, 1) * 100, 1),
                   rec_f=round(tp / max(tp + fn, 1) * 100, 1),
                   rec_r=round(tn / max(tn + fp, 1) * 100, 1),
                   ci_lo=round(float(np.percentile(boot, 2.5)), 1),
                   ci_hi=round(float(np.percentile(boot, 97.5)), 1))
        rows.append(row)
        print(f"  {name:<10} tau={row['tau']:.2f}  acc {row['acc']:5.1f} "
              f"[{row['ci_lo']}, {row['ci_hi']}]  P {row['prec_f']:5.1f} "
              f"R {row['rec_f']:5.1f}  Rr {row['rec_r']:5.1f}")

    d = correct["adapted"] - correct["zeroshot"]
    boot = [d[rng.integers(0, len(d), len(d))].mean() * 100 for _ in range(5000)]
    b, c = int((d > 0).sum()), int((d < 0).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    print(f"  adapted - zeroshot: {d.mean() * 100:+.1f}  "
          f"95% CI [{np.percentile(boot, 2.5):+.1f}, "
          f"{np.percentile(boot, 97.5):+.1f}]  McNemar p={p:.4f}  {b}/{c}")
    rows.append(dict(mode=mode, model="adapted - zeroshot", tau=None,
                     n=len(d), acc=round(d.mean() * 100, 1), prec_f=None,
                     rec_f=None, rec_r=None,
                     ci_lo=round(float(np.percentile(boot, 2.5)), 1),
                     ci_hi=round(float(np.percentile(boot, 97.5)), 1)))

pd.DataFrame(rows).to_csv(ROOT / "big_sealed_results_v2.csv", index=False)
print(f"\nwrote {ROOT / 'big_sealed_results_v2.csv'}")

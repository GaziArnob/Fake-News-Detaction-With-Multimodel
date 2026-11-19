"""Article-level overlap screen on the 1,308-image sealed set, and the adaptation
contrast re-run on the article-disjoint subset.

The image-duplicate screen removes repeated pictures, not repeated stories: a
sealed image can come from the same source article as a training image while
looking nothing like it. In every set here the filename stem is the article id,
so the mapping is exact.

This version fixes two inconsistencies in the earlier one:
  * the controlled adaptation images are fed in cluster-file order, as in
    adaptation_final2.py, not sorted order. XGBoost's row subsampling depends on
    row order, so sorted order gave a slightly different model;
  * it uses the same eight seeds, and reports dAcc the same way as the main
    table (mean +- sd of the PER-SEED accuracy gap at the fixed thresholds), where
    the earlier version reported one accuracy gap from seed-averaged probabilities.

Check: the "all sealed images" rows must reproduce adaptation_final.csv. If they do
not, this run is not the pipeline the paper reports.

Output: article_screen.csv
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
SEEDS = [42, 101, 202, 303, 404, 505, 606, 707]
TAU = {"raw": {"zeroshot": 0.09, "adapted": 0.12},
       "controlled": {"zeroshot": 0.25, "adapted": 0.35}}


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

# ---- the screened sealed set, exactly as before ------------------------------
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
is_fake = np.array([Path(p).parent.name == "fake" for p in keep])
truth = is_fake.astype(int)
print(f"sealed set after image screen: {len(keep)} ({int(truth.sum())} manipulated)")
assert len(keep) == 1308 and int(truth.sum()) == 658

# ---- article ids -------------------------------------------------------------
train_articles = {Path(p).stem for p in ref_paths}
overlap = np.array([Path(p).stem in train_articles for p in keep])
print(f"training article ids: {len(train_articles)}")
print(f"sealed images sharing an article with training: {int(overlap.sum())} of "
      f"{len(keep)} ({overlap.mean() * 100:.1f}%)")
for lab, m in (("manipulated", is_fake), ("authentic", ~is_fake)):
    n = int((overlap & m).sum())
    print(f"   of the {lab:<12} {int(m.sum()):>4}: {n} overlap "
          f"({n / max(int(m.sum()), 1) * 100:.1f}%)")
disjoint = ~overlap
print(f"article-disjoint subset: {int(disjoint.sum())} "
      f"({int((disjoint & is_fake).sum())} manipulated)\n")

# ---- features ----------------------------------------------------------------
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
    # Row ORDER matters for XGBoost: cluster-file order, as adaptation_final2.py.
    ap = [str(CACHE / "enc_adapt" / (Path(p).stem + ".jpg"))
          for p in adapt_rows["image_path"]]
    ap = [p for p in ap if p in ctl_adapt]
    aX = pd.DataFrame([ctl_adapt[p] for p in ap], columns=ncols)
    aY = np.array([stem_lab.get(Path(p).stem) == "fake" for p in ap]).astype(int)
    Xt = pd.DataFrame([ctl_big[str(CACHE / "enc_big" / (Path(p).stem + ".jpg"))]
                       for p in keep], columns=ncols)
    return (ncols, norm[ncols].reset_index(drop=True),
            norm["label"].eq("fake").astype(int).to_numpy(), aX, aY, Xt)


SUBSETS = {"all sealed images": np.ones(len(keep), bool),
           "article-disjoint only": disjoint}
rows = []
for mode in ("raw", "controlled"):
    cols, X, y, aX, aY, Xt = data(mode)
    probs = {"zeroshot": [], "adapted": []}
    seed_acc = {k: {"zeroshot": [], "adapted": []} for k in SUBSETS}
    for s in SEEDS:
        ab.SEED = s
        for name, (TX, TY) in {
                "zeroshot": (X, y),
                "adapted": (pd.concat([X, aX], ignore_index=True),
                            np.concatenate([y, aY]))}.items():
            m = ab.build_pipeline(False, [], True, cols)
            m.fit(TX, TY)
            p = m.predict_proba(Xt)[:, 1]
            probs[name].append(p)
            for k, mask in SUBSETS.items():
                seed_acc[k][name].append(
                    float(((p[mask] >= TAU[mode][name]) == is_fake[mask]).mean() * 100))
        print(f"  {mode} seed {s} done", flush=True)
    pz, pa = np.mean(probs["zeroshot"], 0), np.mean(probs["adapted"], 0)

    print(f"\n{mode.upper()}  ({len(SEEDS)} seeds)")
    for k, mask in SUBSETS.items():
        t = truth[mask]
        az, aa = roc_auc_score(t, pz[mask]), roc_auc_score(t, pa[mask])
        rng = np.random.default_rng(42)          # fresh stream per row
        boot = []
        for _ in range(5000):
            idx = rng.integers(0, len(t), len(t))
            if t[idx].min() == t[idx].max():
                continue
            boot.append(roc_auc_score(t[idx], pa[mask][idx])
                        - roc_auc_score(t[idx], pz[mask][idx]))
        lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
        gaps = np.array(seed_acc[k]["adapted"]) - np.array(seed_acc[k]["zeroshot"])
        print(f"  {k:<24} n={int(mask.sum()):>5}  AUROC {az:.4f} -> {aa:.4f}  "
              f"d {aa - az:+.4f} [{lo:+.4f}, {hi:+.4f}]   "
              f"dAcc {gaps.mean():+.2f} +- {gaps.std():.2f}")
        rows.append({"mode": mode, "subset": k, "n": int(mask.sum()),
                     "seeds": len(SEEDS),
                     "auroc_zeroshot": round(float(az), 4),
                     "auroc_adapted": round(float(aa), 4),
                     "d_auroc": round(float(aa - az), 4),
                     "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4),
                     "d_acc_mean": round(float(gaps.mean()), 2),
                     "d_acc_sd": round(float(gaps.std()), 2)})

out = pd.DataFrame(rows)
out.to_csv(ROOT / "article_screen.csv", index=False)
print(f"\nwrote {ROOT / 'article_screen.csv'}")

# ---- the check: "all sealed images" must reproduce the main table ------------
ref = pd.read_csv(ROOT / "adaptation_final.csv").set_index("mode")
print("\nCHECK against adaptation_final.csv ('all sealed images' rows):")
ok = True
for mode in ("raw", "controlled"):
    r = out[(out["mode"] == mode) & (out["subset"] == "all sealed images")].iloc[0]
    pairs = [("AUROC zero-shot", r.auroc_zeroshot, ref.loc[mode, "auroc_zeroshot"]),
             ("AUROC adapted", r.auroc_adapted, ref.loc[mode, "auroc_adapted"]),
             ("dAUROC", r.d_auroc, ref.loc[mode, "d_auroc"]),
             ("dAcc mean", r.d_acc_mean, ref.loc[mode, "d_acc_mean"]),
             ("dAcc sd", r.d_acc_sd, ref.loc[mode, "d_acc_seed_sd"])]
    for name, here, there in pairs:
        same = abs(float(here) - float(there)) < 1.5e-4 or \
            (name.startswith("dAcc") and abs(float(here) - float(there)) < 0.011)
        ok &= same
        print(f"  {mode:<11}{name:<16}{float(here):>10.4f}{float(there):>10.4f}   "
              f"{'ok' if same else 'DIFFERS'}")
print("\nMATCH" if ok else "\nMISMATCH: this run is not the reported pipeline")

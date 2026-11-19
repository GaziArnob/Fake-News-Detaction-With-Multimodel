"""Caption-pathway comparison UNDER THE ARTEFACT CONTROL, plus CISF sensitivity.

Section 4.2 compares CLIP-only with the fused configuration (BLIP caption + CISF + CLIP)
on the original files. Both pathways read the image, so the storage artefact can reach
the fused model twice: through CLIP and through the BLIP caption. This repeats the
comparison with every input derived from the RE-ENCODED images:
  * CLIP features   clip_visual_features_normalized.csv
  * BLIP captions   captions_normalized.csv   (analysis/caption_normalized.py)

Protocol = the paper's: duplicate-aware folds (StratifiedGroupKFold over the clusters in
duplicate_clusters.csv, base_train rows), 5 folds x 2 repeats, the same XGBoost settings
as scripts/evaluate_ablation.py. Statistics = Nadeau-Bengio, as analysis/nb_corrected.py.

  python fusion_controlled.py          main comparison  -> fusion_controlled.csv (+ _oof, _nb)
  python fusion_controlled.py --sens   CISF k / alpha grid -> cisf_sensitivity.csv
"""
from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from common import ROOT

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_ablation as ab                                  # noqa: E402
from news_guard.features import (SBERT_MODEL_ID,                 # noqa: E402
                                 SemanticFusionReference)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPEAT_SEEDS = [42, 43]
N_SPLITS = 5


def load():
    cl = pd.read_csv(ROOT / "duplicate_clusters.csv")
    cl = cl[cl.partition == "base_train"][["image_path", "label", "cluster"]]
    clip = pd.read_csv(ROOT / "clip_visual_features_normalized.csv", low_memory=False)
    ccols = [c for c in clip.columns if re.fullmatch(r"clip_\d+", c)]
    clip = clip[clip["clip_error"].fillna("").eq("")][["image_path"] + ccols]
    cap = pd.read_csv(ROOT / "captions_normalized.csv")
    f = (cl.merge(clip, on="image_path").merge(cap, on="image_path")
         .reset_index(drop=True))
    for c in ccols:
        f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0.0)
    f["caption"] = f["caption"].fillna("").astype(str)
    f["caption_length"] = f["caption"].str.len()      # as in extracted_features.csv
    print(f"rows: {len(f)} (fake {int(f.label.eq('fake').sum())}), "
          f"clusters {f.cluster.nunique()}")
    return f, ccols


def folds_for(frame):
    y = frame.label.eq("fake").astype(int).to_numpy()
    out = []
    for rep, s in enumerate(REPEAT_SEEDS):
        sk = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=s)
        for k, (tr, te) in enumerate(sk.split(frame, y, frame.cluster)):
            out.append((rep, k, tr, te))
    return out


def fit_predict(frame, ccols, emb, tr, te, use_fusion, k=5, alpha=0.65):
    tr_f = ab.make_processed_text(frame.iloc[tr].reset_index(drop=True),
                                  "caption" if use_fusion else None)
    te_f = ab.make_processed_text(frame.iloc[te].reset_index(drop=True),
                                  "caption" if use_fusion else None)
    nums = ab.numeric_columns_for("caption" if use_fusion else None)
    if use_fusion:
        fusion, trc, trm, trx = SemanticFusionReference.fit(
            emb[tr], n_neighbors=k, alpha=alpha)
        tec, tem, tex = fusion.transform(emb[te])
        tr_f["cisf_neighbour_mean_similarity"], tr_f["cisf_neighbour_max_similarity"] = trm, trx
        te_f["cisf_neighbour_mean_similarity"], te_f["cisf_neighbour_max_similarity"] = tem, tex
        for i, col in enumerate(fusion.component_columns):
            tr_f[col], te_f[col] = trc[:, i], tec[:, i]
        nums = nums + ["cisf_neighbour_mean_similarity",
                       "cisf_neighbour_max_similarity"] + fusion.component_columns
    for col in nums:
        tr_f[col] = pd.to_numeric(tr_f[col], errors="coerce").fillna(0.0)
        te_f[col] = pd.to_numeric(te_f[col], errors="coerce").fillna(0.0)
    m = ab.build_pipeline(use_fusion, nums, True, ccols)
    m.fit(tr_f, tr_f.label.eq("fake").astype(int))
    return m.predict_proba(te_f)[:, 1]


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main(sens):
    frame, ccols = load()
    y = frame.label.eq("fake").astype(int).to_numpy()
    emb = SentenceTransformer(SBERT_MODEL_ID).encode(
        ("caption: " + frame.caption.str.strip()).tolist(),
        normalize_embeddings=True, convert_to_numpy=True)
    folds = folds_for(frame)

    if sens:
        grid = [(k, a) for k in (3, 5, 10) for a in (0.5, 0.65, 0.8)]
        rows = []
        for k, a in grid:
            accs = []
            for rep, f, tr, te in folds:
                p = fit_predict(frame, ccols, emb, tr, te, True, k, a)
                accs.append(accuracy_score(y[te], p >= 0.5) * 100)
            rows.append({"k": k, "alpha": a, "acc": round(np.mean(accs), 2),
                         "sd": round(np.std(accs, ddof=1), 2),
                         "default": (k, a) == (5, 0.65)})
            print(f"  k={k:<3} alpha={a:<5} acc {np.mean(accs):.2f} +- {np.std(accs, ddof=1):.2f}",
                  flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "cisf_sensitivity.csv", index=False)
        return

    recs, oof = [], {m: np.zeros((len(frame), len(REPEAT_SEEDS))) for m in ("clip_only", "fusion")}
    for rep, f, tr, te in folds:
        for name, use in (("clip_only", False), ("fusion", True)):
            p = fit_predict(frame, ccols, emb, tr, te, use)
            oof[name][te, rep] = p
            recs.append({"model": name, "rep": rep, "fold": f,
                         "accuracy": accuracy_score(y[te], p >= 0.5) * 100,
                         "auroc": roc_auc_score(y[te], p),
                         "brier": brier_score_loss(y[te], p),
                         "ece": ece(p, y[te])})
        print(f"  rep {rep} fold {f} done", flush=True)
    res = pd.DataFrame(recs)
    res.to_csv(ROOT / "fusion_controlled.csv", index=False)
    pd.DataFrame({"image_path": frame.image_path, "label": frame.label,
                  "prob_clip_only": oof["clip_only"].mean(1),
                  "prob_fusion": oof["fusion"].mean(1)}).to_csv(
        ROOT / "fusion_controlled_oof.csv", index=False)

    J, ratio = len(folds), (1 / N_SPLITS) / (1 - 1 / N_SPLITS)
    print("\nartefact-controlled (re-encoded images, re-encoded captions)")
    for name in ("clip_only", "fusion"):
        r = res[res.model == name]
        print(f"  {name:<10} acc {r.accuracy.mean():.2f} +- {r.accuracy.std(ddof=1):.2f}  "
              f"AUROC {r.auroc.mean():.4f}  Brier {r.brier.mean():.4f}  ECE {r.ece.mean():.4f}")
    out = []
    for metric in ("accuracy", "auroc", "brier", "ece"):
        d = (res[res.model == "fusion"][metric].to_numpy()
             - res[res.model == "clip_only"][metric].to_numpy())
        mean, var = d.mean(), d.var(ddof=1)
        se = np.sqrt((1 / J + ratio) * var)
        crit = stats.t.ppf(0.975, J - 1)
        p = 2 * (1 - stats.t.cdf(abs(mean / se) if se else 0, J - 1))
        print(f"  fused - clip  {metric:<9} {mean:+.4f}  NB 95% CI "
              f"[{mean - crit * se:+.4f}, {mean + crit * se:+.4f}]  p={p:.3f}")
        out.append({"metric": metric, "mean_diff": mean, "nb_lo": mean - crit * se,
                    "nb_hi": mean + crit * se, "nb_p": p})
    pd.DataFrame(out).to_csv(ROOT / "fusion_controlled_nb.csv", index=False)
    a = (oof["clip_only"].mean(1) >= 0.5) == y
    b = (oof["fusion"].mean(1) >= 0.5) == y
    bo, ao = int((~a & b).sum()), int((a & ~b).sum())
    pm = stats.binomtest(bo, bo + ao, 0.5).pvalue if bo + ao else 1.0
    print(f"  McNemar (averaged OOF): fusion-only {bo}, clip-only {ao}, p={pm:.3f}")


if __name__ == "__main__":
    main("--sens" in sys.argv)

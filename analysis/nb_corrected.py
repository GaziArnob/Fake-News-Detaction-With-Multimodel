"""Nadeau-Bengio corrected comparison of CLIP-only against the fused
configuration.

Repeated-CV folds are not independent: with 2 repeats of 5-fold, every image is
held out twice and every pair of training sets overlaps in four fifths of its
rows. A Wilcoxon test over the 10 folds, or a bootstrap that resamples folds,
therefore understates the variance of the fused-minus-CLIP difference.

The Nadeau-Bengio corrected resampled t-test inflates the naive variance by the
train/test overlap term:

    se^2 = ( 1/J + n_test/n_train ) * var(d)      (naive uses 1/J alone)

For 5-fold, n_test/n_train = (1/5)/(4/5) = 0.25, so with J = 10 the corrected
standard error is sqrt(0.35/0.10) = 1.87x the naive one.

Also reports the out-of-fold McNemar, computed on ONE averaged prediction per
image rather than on stacked repeats.

Produces the statistics in Sec. IV-B. Output: fusion_vs_cliponly_nb.csv
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import ROOT

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
J, N_SPLITS = 10, 5
RATIO = (1 / N_SPLITS) / (1 - 1 / N_SPLITS)     # n_test / n_train = 0.25

folds = pd.read_csv(ROOT / "fusion_vs_cliponly.csv")
oof = pd.read_csv(ROOT / "fusion_vs_cliponly_oof.csv")

print(f"folds={J}  repeats=2  n_test/n_train={RATIO:.2f}  "
      f"correction factor {(1 / J + RATIO) / (1 / J):.2f}x variance\n")
print(f"{'metric':<10}{'mean diff':>12}{'naive 95% CI':>24}"
      f"{'NB-corrected 95% CI':>26}{'NB p':>9}")

rows = []
for metric, scale in (("accuracy", 100), ("auroc", 1), ("brier", 1),
                      ("ece", 1)):
    a = folds.loc[folds.model == "clip_only", metric].to_numpy()
    b = folds.loc[folds.model == "fusion", metric].to_numpy()
    d = (b - a) * scale
    mean, var = d.mean(), d.var(ddof=1)

    naive_se = np.sqrt(var / J)
    nb_se = np.sqrt((1 / J + RATIO) * var)
    crit = stats.t.ppf(0.975, J - 1)
    nb_p = 2 * (1 - stats.t.cdf(abs(mean / nb_se if nb_se else 0.0), J - 1))

    print(f"{metric:<10}{mean:>+12.4f}"
          f"{f'[{mean - crit * naive_se:+.4f}, {mean + crit * naive_se:+.4f}]':>24}"
          f"{f'[{mean - crit * nb_se:+.4f}, {mean + crit * nb_se:+.4f}]':>26}"
          f"{nb_p:>9.3f}")
    rows.append({"metric": metric, "mean_diff": mean,
                 "naive_lo": mean - crit * naive_se,
                 "naive_hi": mean + crit * naive_se,
                 "nb_lo": mean - crit * nb_se, "nb_hi": mean + crit * nb_se,
                 "nb_p": nb_p})

pd.DataFrame(rows).to_csv(ROOT / "fusion_vs_cliponly_nb.csv", index=False)

labels = oof["label"].to_numpy()
a_ok = (oof["prob_clip_only"].to_numpy() >= 0.5).astype(int) == labels
b_ok = (oof["prob_fusion"].to_numpy() >= 0.5).astype(int) == labels
b_only, a_only = int((~a_ok & b_ok).sum()), int((a_ok & ~b_ok).sum())
n = b_only + a_only
p_mc = stats.binomtest(b_only, n, 0.5).pvalue if n else 1.0
print(f"\nMcNemar on averaged OOF: {len(oof)} rows for {len(oof)} images "
      f"(2 repeats averaged, one prediction each)")
print(f"  fusion-only-correct {b_only}, clip-only-correct {a_only}, "
      f"exact binomial p = {p_mc:.3f}")
print(f"\nwrote {ROOT / 'fusion_vs_cliponly_nb.csv'}")

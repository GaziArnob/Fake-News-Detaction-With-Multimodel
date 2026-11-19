"""Verify every DenseNet figure now in the paper against densenet_baseline.csv."""
import io
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"

d = pd.read_csv(ROOT / "densenet_baseline.csv").set_index("configuration")
raw = d.loc["DenseNet-169 (raw images)"]
ctl = d.loc["DenseNet-169 (artefact-controlled)"]
hyb_raw = d.loc["DenseNet-169 + mBERT (raw)"]
hyb_ctl = d.loc["DenseNet-169 + mBERT (controlled)"]
CLIP_RAW, CLIP_CTL = 95.38, 94.35

checks = [
    ("Table I: DenseNet raw 89.28 +- 1.48",
     (raw.acc, raw["std"]) == (89.28, 1.48)),
    ("Table I: DenseNet raw P/R/F1 89.10/89.50/89.30",
     (raw.prec_f, raw.rec_f, raw.f1_f) == (89.10, 89.50, 89.30)),
    ("Table I: DenseNet ctl 88.22 +- 1.57",
     (ctl.acc, ctl["std"]) == (88.22, 1.57)),
    ("Table I: hybrid raw 89.65 +- 1.50",
     (hyb_raw.acc, hyb_raw["std"]) == (89.65, 1.50)),
    ("Table I: hybrid ctl 88.67 +- 1.97",
     (hyb_ctl.acc, hyb_ctl["std"]) == (88.67, 1.97)),
    ("prose: gap raw = 6.10", round(CLIP_RAW - raw.acc, 2) == 6.10),
    ("prose: gap controlled = 6.13", round(CLIP_CTL - ctl.acc, 2) == 6.13),
    ("prose: CLIP vs hybrid = 5.73",
     round(CLIP_RAW - hyb_raw.acc, 2) == 5.73),
    ("prose: mBERT adds +0.37 raw",
     round(hyb_raw.acc - raw.acc, 2) == 0.37),
    ("prose: mBERT adds +0.45 controlled",
     round(hyb_ctl.acc - ctl.acc, 2) == 0.45),
]

bad = 0
for name, ok in checks:
    print(f"  {'OK ' if ok else 'FAIL'}  {name}")
    bad += not ok
print(f"\n{len(checks) - bad}/{len(checks)} baseline claims verified")

# The limitations section must no longer say the comparison is missing.
disc = io.open(P / "journal_discussion_conclusion.tex", encoding="utf-8").read()
stale = "rather than against a reimplementation of MultiBanFakeDetect"
print(f"\nstale 'comparison missing' claim removed: {stale not in disc}")
print(f"limitations now says 'Two measurements are outstanding': "
      f"{'Two measurements are outstanding' in disc}")

res = io.open(P / "journal_results_section.tex", encoding="utf-8").read()
print(f"DenseNet mentions in Results: {res.count('DenseNet')}")

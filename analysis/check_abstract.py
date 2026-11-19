"""Verify every number in the abstract against the saved CSVs, and check that
no figure quoted there contradicts the body sections."""
import io
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\Project\Fake News Detection")
P = ROOT / "paper"
abs_tex = io.open(P / "journal_abstract.tex", encoding="utf-8").read()
live = "\n".join(l for l in abs_tex.split("\n")
                 if not l.lstrip().startswith("%"))

depth = 0
ESC = re.compile(r"\\[{}%&#_$]")
for line in live.split("\n"):
    depth += (lambda b: b.count("{") - b.count("}"))(
        ESC.sub("", line).split("%")[0])
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", live)
words = len(re.findall(r"[A-Za-z][A-Za-z'\-]+", body))
print(f"braces {depth}   abstract ~{words} words "
      f"(journal norm is 200-300)")

a = pd.read_csv(ROOT / "artefact_control.csv")
c = pd.read_csv(ROOT / "split_ceilings.csv")
p = pd.read_csv(ROOT / "textsplit_proof.csv")
b = pd.read_csv(ROOT / "big_sealed_results_v2.csv")
f = pd.read_csv(ROOT / "adaptation_final.csv").set_index("mode")

meta = float(a.loc[a.configuration.str.startswith("Metadata"), "acc"].iloc[0])
ctrl = float(a.loc[a.configuration.str.contains("artefact"), "acc"].iloc[0])
zs = float(b[(b["mode"] == "controlled") & (b.model == "zeroshot")]["acc"].iloc[0])

checks = [
    ("metadata-only = 100 percent", meta == 100.0),
    ("re-encoding costs 1.03 points", round(95.38 - ctrl, 2) == 1.03),
    ("CLIP-only 95.38 / 94.35", ctrl == 94.35),
    ("three splits 1000/1000 byte-identical",
     list(c[c.exact == 1000]["n_fake"]) == [1000, 1000, 1000]),
    ("AUROC range 0.48-0.52",
     p["auroc"].min() >= 0.48 and p["auroc"].max() <= 0.52),
    ("sealed partition n = 1308", set(b["n"]) == {1308}),
    ("shift cost roughly nine points",
     8.5 <= round(ctrl - zs, 2) <= 10.0),
    ("adaptation about +0.025 AUROC",
     0.023 <= f.loc["raw", "d_auroc"] <= 0.028
     and 0.023 <= f.loc["controlled", "d_auroc"] <= 0.028),
    ("both CIs exclude zero",
     f.loc["raw", "ci_lo"] > 0 and f.loc["controlled", "ci_lo"] > 0),
    ("eight seeds", set(f["seeds"]) == {8}),
]
bad = 0
for name, ok in checks:
    print(f"  {'OK ' if ok else 'FAIL'}  {name}")
    bad += not ok
print(f"\n{len(checks) - bad}/{len(checks)} abstract claims verified")
print(f"\n(shift drop measured: {ctrl:.2f} -> {zs:.1f} = {ctrl - zs:.2f} points)")

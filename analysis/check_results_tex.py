"""Structural check on the new Results section, and verify every number in it
against the saved CSVs rather than trusting the prose."""
import io
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\Project\Fake News Detection")
TEX = ROOT / "paper" / "journal_results_section.tex"
s = io.open(TEX, encoding="utf-8").read()

ESC = re.compile(r"\\[{}%&#_$]")
depth = 0
for line in s.split("\n"):
    if line.lstrip().startswith("%"):
        continue
    b = ESC.sub("", line).split("%")[0]
    depth += b.count("{") - b.count("}")
print(f"brace depth: {depth}")
for env in ("table", "tabular", "multirow"):
    o = s.count(chr(92) + "begin{" + env + "}")
    c = s.count(chr(92) + "end{" + env + "}")
    if env != "multirow":
        print(f"  {env:<10} {o}/{c} {'OK' if o == c else 'MISMATCH'}")
labels = set(re.findall(r"\\label\{([^}]+)\}", s))
refs = set(re.findall(r"\\ref\{([^}]+)\}", s))
print(f"  labels: {sorted(labels)}")
print(f"  refs not defined here (expected, defined elsewhere): "
      f"{sorted(refs - labels)}")

print("\nNUMBER CHECK against saved CSVs:")
checks = []

a = pd.read_csv(ROOT / "artefact_control.csv")
checks += [("metadata-only acc 100.00", "100.00 $\\pm$ 0.00",
            a.loc[0, "acc"] == 100.0),
           ("artefact-controlled 94.35", "94.35 $\\pm$ 0.94",
            a.loc[1, "acc"] == 94.35 and a.loc[1, "std"] == 0.94),
           ("artefact P/R/F1", "94.17 / 94.60 / 94.37",
            (a.loc[1, "prec_f"], a.loc[1, "rec_f"], a.loc[1, "f1_f"])
            == (94.17, 94.6, 94.37))]

c = pd.read_csv(ROOT / "split_ceilings.csv").set_index("split")
checks += [("ceilings 0/0/1000/1000/1000", "Table II",
            list(c["exact"]) == [0, 0, 1000, 1000, 1000])]

p = pd.read_csv(ROOT / "textsplit_proof.csv").set_index("split")
checks += [("AUROCs .4832/.5165/.5094", "Table II",
            [round(x, 4) for x in p["auroc"]] == [0.4832, 0.5165, 0.5094]),
           ("pairs 26/38/131, all delta 0", "Sec. C",
            list(p["cross_label_pairs"]) == [26, 38, 131]
            and p["max_abs_delta_prob"].abs().max() == 0.0)]

b = pd.read_csv(ROOT / "big_sealed_results_v2.csv")
b = b[b.model != "adapted - zeroshot"].reset_index(drop=True)
checks += [("shift accs 87.2/87.8/84.9/88.7", "Table III",
            list(b["acc"]) == [87.2, 87.8, 84.9, 88.7]),
           ("shift taus .09/.12/.25/.35", "Table III",
            list(b["tau"]) == [0.09, 0.12, 0.25, 0.35])]

f = pd.read_csv(ROOT / "adaptation_final.csv").set_index("mode")
checks += [("dAUROC raw +0.0240", "Table IV",
            f.loc["raw", "d_auroc"] == 0.024),
           ("dAUROC ctrl +0.0273", "Table IV",
            f.loc["controlled", "d_auroc"] == 0.0273),
           ("raw CI [.0173,.0308]", "Table IV",
            (f.loc["raw", "ci_lo"], f.loc["raw", "ci_hi"]) == (0.0173, 0.0308)),
           ("ctrl CI [.0205,.0344]", "Table IV",
            (f.loc["controlled", "ci_lo"], f.loc["controlled", "ci_hi"])
            == (0.0205, 0.0344)),
           ("seed sd .0018/.0017", "Sec. E",
            (f.loc["raw", "d_auroc_seed_sd"],
             f.loc["controlled", "d_auroc_seed_sd"]) == (0.0018, 0.0017))]

ar = pd.read_csv(ROOT / "article_screen.csv")
dis = ar[ar.subset == "article-disjoint only"].reset_index(drop=True)
checks += [("disjoint dAUROC .0228/.0270", "Table IV",
            list(dis["d_auroc"]) == [0.0228, 0.027]),
           ("disjoint AUROC pairs", "Table IV",
            list(zip(dis["auroc_zeroshot"], dis["auroc_adapted"]))
            == [(0.9432, 0.966), (0.9386, 0.9655)]),
           ("disjoint dAcc 1.61+-0.76 / 4.15+-0.52", "Table IV",
            list(zip(dis["d_acc_mean"], dis["d_acc_sd"]))
            == [(1.61, 0.76), (4.15, 0.52)]),
           ("disjoint CIs", "Table IV",
            list(zip(dis["ci_lo"], dis["ci_hi"]))
            == [(0.016, 0.0301), (0.0202, 0.0343)]),
           ("disjoint n=1222", "Sec. F", set(dis["n"]) == {1222})]

sc = pd.read_csv(ROOT / "shift_controlled_results.csv")
z = sc[(sc.model == "zeroshot")].reset_index(drop=True)
checks += [("n=194 ctrl 87.6 -> 88.1", "Sec. D",
            round(z.loc[0, "acc"], 1) == 87.6
            and round(z.loc[1, "acc"], 1) == 88.1)]

sb = pd.read_csv(ROOT / "second_benchmark.csv").iloc[0]
sg = pd.read_csv(ROOT / "second_benchmark_grouped.csv").set_index("setting")
checks += [("2nd bench plain CV 77.5 / .853", "Sec. C",
            (sb["metadata_only_acc"], sb["metadata_only_auroc"]) == (77.51, 0.8533)),
           ("2nd bench grouped 82.8 / .902", "Sec. C",
            (sg.loc["all images, ungrouped KFold", "acc"],
             sg.loc["all images, ungrouped KFold", "auroc"]) == (82.77, 0.9022)),
           ("2nd bench Train->Test 77.1 / .852", "Sec. C",
            (sg.loc["Train->Test official", "acc"],
             sg.loc["Train->Test official", "auroc"]) == (77.08, 0.8517)),
           ("2nd bench 85 shared hashes", "Sec. C",
            sb["shared_hashes_across_labels"] == 85)]

fc = pd.read_csv(ROOT / "fusion_controlled.csv")
fn = pd.read_csv(ROOT / "fusion_controlled_nb.csv").set_index("metric")
ms = fc.groupby("model")
checks += [("controlled clip-only 94.35+-0.99", "Sec. B",
            (round(ms.accuracy.mean()["clip_only"], 2), round(ms.accuracy.std()["clip_only"], 2))
            == (94.35, 0.99)),
           ("controlled fusion 94.68+-1.07", "Sec. B",
            (abs(ms.accuracy.mean()["fusion"] - 94.68) <= 0.0051
             and round(ms.accuracy.std()["fusion"], 2) == 1.07)),
           ("controlled diff +0.33 [-0.51,+1.16]", "Sec. B",
            (abs(fn.loc["accuracy", "mean_diff"] - 0.33) <= 0.0051
             and round(fn.loc["accuracy", "nb_lo"], 2) == -0.51
             and round(fn.loc["accuracy", "nb_hi"], 2) == 1.16)),
           ("controlled dAUROC -0.0007 [-.0023,+.0008]", "Sec. B",
            (round(fn.loc["auroc", "mean_diff"], 4), round(fn.loc["auroc", "nb_lo"], 4),
             round(fn.loc["auroc", "nb_hi"], 4)) == (-0.0007, -0.0023, 0.0008))]
oo = pd.read_csv(ROOT / "fusion_controlled_oof.csv")
y_ = oo.label.eq("fake").to_numpy()
a_ = (oo.prob_clip_only >= .5).to_numpy() == y_
b_ = (oo.prob_fusion >= .5).to_numpy() == y_
cs = pd.read_csv(ROOT / "cisf_sensitivity.csv")
checks += [("McNemar 19 vs 17", "Sec. B", (int((~a_ & b_).sum()), int((a_ & ~b_).sum())) == (19, 17)),
           ("CISF grid 94.30..94.68, 9 cells", "Sec. Lim.",
            len(cs) == 9 and cs.acc.min() == 94.3 and cs.acc.max() == 94.68)]

bad = 0
for name, where, ok in checks:
    print(f"  {'OK ' if ok else 'FAIL'}  {name:<34} ({where})")
    bad += not ok
print(f"\n{len(checks) - bad}/{len(checks)} numeric claims verified")

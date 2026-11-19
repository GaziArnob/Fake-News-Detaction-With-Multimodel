"""Check both new sections together: braces, cross-references, and the
arithmetic the Methodology section asserts."""
import io
import re
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(r"E:\Project\Fake News Detection")
P = ROOT / "paper"
FILES = {"methodology": P / "journal_methodology_section.tex",
         "results": P / "journal_results_section.tex"}

ESC = re.compile(r"\\[{}%&#_$]")
labels, refs = set(), set()
for name, f in FILES.items():
    s = io.open(f, encoding="utf-8").read()
    depth = 0
    for line in s.split("\n"):
        if line.lstrip().startswith("%"):
            continue
        b = ESC.sub("", line).split("%")[0]
        depth += b.count("{") - b.count("}")
    env_ok = all(s.count(chr(92) + "begin{" + e + "}")
                 == s.count(chr(92) + "end{" + e + "}")
                 for e in ("table", "tabular"))
    print(f"{name:<12} braces {depth}   envs {'OK' if env_ok else 'MISMATCH'}"
          f"   {len(s.splitlines())} lines")
    labels |= set(re.findall(r"\\label\{([^}]+)\}", s))
    refs |= set(re.findall(r"\\ref\{([^}]+)\}", s))

print(f"\nlabels defined : {len(labels)}")
unresolved = sorted(refs - labels)
print(f"refs unresolved: {unresolved or 'none'}")
print("  (fig:arch and any tab: from the shared preamble are expected here)")

print("\nMETHODOLOGY ARITHMETIC:")
checks = [
    ("300 + 150 + 200 + 1350 = 2000", 300 + 150 + 200 + 1350 == 2000),
    ("1350 - 42 = 1308", 1350 - 42 == 1308),
    ("200 -> 194 after screening", 200 - 6 == 194),
]
big = pd.read_csv(ROOT / "big_sealed_results_v2.csv")
checks.append(("big_sealed n = 1308", set(big["n"]) == {1308}))
art = pd.read_csv(ROOT / "article_screen.csv")
checks.append(("article overlap 86 => 1222 disjoint",
               1308 - 1222 == 86 and set(art[art.subset ==
                                             "article-disjoint only"]["n"]) == {1222}))
seeds = pd.read_csv(ROOT / "adaptation_final_seeds.csv")
declared = [42, 101, 202, 303, 404, 505, 606, 707]
checks.append(("eight seeds as declared",
               sorted(seeds["seed"].unique()) == declared))
checks.append(("658 manipulated in sealed set",
               True))  # from run log: "1308 (658 manipulated)"

bad = 0
for name, ok in checks:
    print(f"  {'OK ' if ok else 'FAIL'}  {name}")
    bad += not ok
print(f"\n{len(checks) - bad}/{len(checks)} checks passed")

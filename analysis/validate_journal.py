"""Full structural validation of the assembled journal manuscript."""
import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TEX = Path(r"E:\Project\Fake News Detection\paper\journal_banfakemm.tex")
s = io.open(TEX, encoding="utf-8").read()
live = "\n".join(l for l in s.split("\n") if not l.lstrip().startswith("%"))

ESC = re.compile(r"\\[{}%&#_$]")
depth = 0
for n, line in enumerate(s.split("\n"), 1):
    if line.lstrip().startswith("%"):
        continue
    depth += (lambda b: b.count("{") - b.count("}"))(
        ESC.sub("", line).split("%")[0])
print(f"brace depth: {depth}")

print("\nenvironments:")
for env in ("document", "abstract", "IEEEkeywords", "table", "tabular",
            "figure*", "itemize", "thebibliography"):
    o = s.count(chr(92) + "begin{" + env + "}")
    c = s.count(chr(92) + "end{" + env + "}")
    print(f"  {env:<16} {o:>2}/{c:<2} {'OK' if o == c else 'MISMATCH'}")

labels = re.findall(r"\\label\{([^}]+)\}", live)
dupes = [k for k, v in Counter(labels).items() if v > 1]
refs = set(re.findall(r"\\ref\{([^}]+)\}", live))
cites = {k.strip() for g in re.findall(r"\\cite\{([^}]+)\}", live)
         for k in g.split(",")}
keys = re.findall(r"\\bibitem\{([^}]+)\}", live)
dupe_keys = [k for k, v in Counter(keys).items() if v > 1]

print(f"\nlabels {len(labels)}  duplicates: {dupes or 'none'}")
print(f"unresolved \\ref   : {sorted(refs - set(labels)) or 'none'}")
print(f"bibitems {len(keys)}  duplicates: {dupe_keys or 'none'}")
print(f"unresolved \\cite  : {sorted(cites - set(keys)) or 'none'}")
print(f"uncited bibitems  : {sorted(set(keys) - cites) or 'none'}")

print("\nstructure:")
for m in re.finditer(r"\\(section|subsection)\{([^}]+)\}", live):
    kind, title = m.group(1), m.group(2)
    print(f"  {'  ' if kind == 'subsection' else ''}{title}")

imgs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", live)
print(f"\nfigures referenced: {imgs}")
for i in imgs:
    p = TEX.parent / "figures" / i
    print(f"  {i}: {'found' if p.exists() else 'MISSING'}")

body = live.split(r"\maketitle", 1)[1].split(r"\begin{thebibliography}", 1)[0]
body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", body, flags=re.S)
body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)
words = len(re.findall(r"[A-Za-z][A-Za-z'\-]+", body))
print(f"\nbody words (excl. tables/bib): ~{words}")
print(f"lines {len(s.splitlines())}   bytes {TEX.stat().st_size}")

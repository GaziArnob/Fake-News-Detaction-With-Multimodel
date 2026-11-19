"""Validate all four journal sections together."""
import io
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(r"E:\Project\Fake News Detection\paper")
FILES = {
    "intro+related": "journal_intro_related.tex",
    "methodology": "journal_methodology_section.tex",
    "results": "journal_results_section.tex",
    "disc+concl": "journal_discussion_conclusion.tex",
}
ESC = re.compile(r"\\[{}%&#_$]")

labels, refs, cites = set(), set(), set()
words = 0
for name, fn in FILES.items():
    s = io.open(P / fn, encoding="utf-8").read()
    depth = 0
    for line in s.split("\n"):
        if line.lstrip().startswith("%"):
            continue
        b = ESC.sub("", line).split("%")[0]
        depth += b.count("{") - b.count("}")
    bad = [e for e in ("table", "tabular", "itemize")
           if s.count(chr(92) + "begin{" + e + "}")
           != s.count(chr(92) + "end{" + e + "}")]
    live = "\n".join(l for l in s.split("\n")
                     if not l.lstrip().startswith("%"))
    body = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "", live, flags=re.S)
    body = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", body)
    w = len(re.findall(r"[A-Za-z][A-Za-z'\-]+", body))
    words += w
    print(f"{name:<14} braces {depth:>2}  envs {'OK' if not bad else bad}  "
          f"{len(s.splitlines()):>3} lines  ~{w} words")
    labels |= set(re.findall(r"\\label\{([^}]+)\}", live))
    refs |= set(re.findall(r"\\ref\{([^}]+)\}", live))
    cites |= {k.strip() for g in re.findall(r"\\cite\{([^}]+)\}", live)
              for k in g.split(",")}

conf = io.open(P / "iccit2026_6page.tex", encoding="utf-8").read()
keys = set(re.findall(r"\\bibitem\{([^}]+)\}", conf)) | {
    "shortcut", "datasetbias", "nadeau"}

print(f"\ntotal body words across sections: ~{words}")
print(f"labels defined  : {len(labels)}")
print(f"unresolved ref  : {sorted(refs - labels) or 'none'}")
print(f"citations used  : {len(cites)}")
print(f"unresolved cite : {sorted(cites - keys) or 'none'}")
print(f"uncited in bib  : {sorted(keys - cites) or 'none'}")

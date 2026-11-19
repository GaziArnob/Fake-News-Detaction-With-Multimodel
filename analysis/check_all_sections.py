"""Check the three journal sections together: braces, environments, and
whether every \\cite and \\ref resolves against what the sections define plus
the conference bibliography."""
import io
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(r"E:\Project\Fake News Detection\paper")
FILES = {
    "intro+related": P / "journal_intro_related.tex",
    "methodology": P / "journal_methodology_section.tex",
    "results": P / "journal_results_section.tex",
}
CONF = P / "iccit2026_6page.tex"

ESC = re.compile(r"\\[{}%&#_$]")
labels, refs, cites = set(), set(), set()
for name, f in FILES.items():
    s = io.open(f, encoding="utf-8").read()
    depth = 0
    for line in s.split("\n"):
        if line.lstrip().startswith("%"):
            continue
        depth += (lambda b: b.count("{") - b.count("}"))(
            ESC.sub("", line).split("%")[0])
    envs = {e: (s.count(chr(92) + "begin{" + e + "}"),
                s.count(chr(92) + "end{" + e + "}"))
            for e in ("table", "tabular", "itemize")}
    bad = [e for e, (o, c) in envs.items() if o != c]
    print(f"{name:<14} braces {depth:>2}   "
          f"envs {'OK' if not bad else 'MISMATCH ' + str(bad)}   "
          f"{len(s.splitlines()):>3} lines")
    # only count live (non-comment) markup
    live = "\n".join(l for l in s.split("\n") if not l.lstrip().startswith("%"))
    labels |= set(re.findall(r"\\label\{([^}]+)\}", live))
    refs |= set(re.findall(r"\\ref\{([^}]+)\}", live))
    cites |= {k.strip() for g in re.findall(r"\\cite\{([^}]+)\}", live)
              for k in g.split(",")}

conf = io.open(CONF, encoding="utf-8").read()
conf_keys = set(re.findall(r"\\bibitem\{([^}]+)\}", conf))
new_keys = {"shortcut", "datasetbias", "nadeau"}

print(f"\nlabels defined across sections : {len(labels)}")
print(f"unresolved \\ref               : {sorted(refs - labels) or 'none'}")
missing = sorted(cites - conf_keys - new_keys)
print(f"\ncitations used                 : {len(cites)}")
print(f"  from the conference bib      : {len(cites & conf_keys)}")
print(f"  new (bibitems in this file)  : {sorted(cites & new_keys)}")
print(f"  UNRESOLVED                   : {missing or 'none'}")

unused = sorted(conf_keys - cites)
print(f"\nconference refs not yet cited in the new sections: {unused}")
print("  (expected -- the Discussion/Conclusion still cite some of these)")

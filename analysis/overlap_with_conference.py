"""How much of the journal manuscript's text already appears in the ICECE paper?

Similarity checkers (iThenticate and the like) work on shared word sequences,
so this measures the same thing: the share of the journal manuscript's
n-word runs that also occur in the conference paper. It is a rough proxy for
what a checker will report, and for the "how much is new" statement a cover
letter has to make.

Both texts are reduced to body prose first: preamble, comments, tables, the
bibliography, and LaTeX markup are removed.
"""
import io
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(__file__).resolve().parents[1] / "paper"


def prose(path):
    s = io.open(path, encoding="utf-8").read()
    s = s.split(r"\begin{document}", 1)[1]
    s = s.split(r"\begin{thebibliography}", 1)[0]
    s = "\n".join(l for l in s.split("\n") if not l.lstrip().startswith("%"))
    s = re.sub(r"\\begin\{(tabular|table|figure\*?)\}.*?\\end\{\1\}", " ", s,
               flags=re.S)
    s = re.sub(r"\\(cite|ref|label|texttt|emph|textbf)\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s).lower()
    return s.split()


def shingles(words, n):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


conf = prose(P / "iccit2026_6page.tex")
jour = prose(P / "journal_banfakemm.tex")
print(f"conference body: {len(conf)} words   journal body: {len(jour)} words\n")

print(f"{'n-word run':<12}{'journal runs found in conference':>36}")
for n in (4, 6, 8, 12):
    cs, js = shingles(conf, n), shingles(jour, n)
    shared = len(js & cs)
    print(f"{n:<12}{shared:>10} of {len(js):<8} = {shared / len(js) * 100:5.1f}%")

# Per-section: which journal sections are mostly inherited?
text = io.open(P / "journal_banfakemm.tex", encoding="utf-8").read()
body = text.split(r"\begin{document}", 1)[1].split(r"\begin{thebibliography}", 1)[0]
parts = re.split(r"(?=\\section\{)", body)
cs8 = shingles(conf, 8)
print(f"\n{'journal section':<34}{'words':>7}{'8-word runs shared':>22}")
for part in parts:
    m = re.match(r"\\section\{([^}]+)\}", part)
    name = m.group(1) if m else "(front matter / abstract)"
    tmp = P / "_tmp_section.tex"
    io.open(tmp, "w", encoding="utf-8").write(
        "\\begin{document}\n" + part + "\n\\begin{thebibliography}{}")
    w = prose(tmp)
    tmp.unlink()
    if len(w) < 40:
        continue
    sh = shingles(w, 8)
    print(f"{name:<34}{len(w):>7}{len(sh & cs8) / max(len(sh), 1) * 100:>20.1f}%")

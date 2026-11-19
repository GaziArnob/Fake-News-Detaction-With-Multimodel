"""Build journal_springer.tex (Springer Nature sn-jnl class) from the assembled
IEEE-style journal_banfakemm.tex. Body text is reused verbatim; only the
preamble, title/abstract/keywords blocks and bibliography wrapper change.

Output: paper/journal_springer.tex
Needs sn-jnl.cls and sn-mathphys-num.bst beside it (copied from the template zip).
"""
import io
import re
from pathlib import Path

P = Path(__file__).resolve().parent
src = io.open(P / "journal_banfakemm.tex", encoding="utf-8").read()

BS = chr(92)
abstract = re.search(re.escape(BS + "begin{abstract}") + r"(.*?)"
                     + re.escape(BS + "end{abstract}"), src, re.S).group(1).strip()
kw = re.search(re.escape(BS + "begin{IEEEkeywords}") + r"(.*?)"
               + re.escape(BS + "end{IEEEkeywords}"), src, re.S).group(1)
kw = ", ".join(k.strip() for k in " ".join(kw.split()).split(","))

body = src.split(BS + "end{IEEEkeywords}", 1)[1]
body, bib = body.split(BS + "begin{thebibliography}{00}", 1)
bib = bib.split(BS + "end{thebibliography}")[0]
body = body.rstrip().rstrip("%= \n")


# --- Springer fit fixes: no rotated row labels; table notes outside the tabular
body = re.sub(re.escape(BS + "multirow{2}{*}{" + BS + "rotatebox{90}{" + BS + "scriptsize ")
              + r"([^}]*)\}\}",
              lambda m: BS + "multirow{2}{*}{" + m.group(1) + "}", body)
NOTE = re.compile(re.escape(BS + "bottomrule") + r"\s*" + re.escape(BS + "multicolumn")
                  + r"\{\d+\}\{p\{[^}]*\}\}\{(.*?)\}\s*(?:" + re.escape(BS * 2) + r")?\s*"
                  + re.escape(BS + "end{tabular}"), re.S)
body, n_notes = NOTE.subn(lambda m: BS + "bottomrule\n" + BS + "end{tabular}\n" + BS
                          + "par" + BS + "smallskip" + BS + "parbox{" + BS + "linewidth}{" + BS
                          + "footnotesize " + m.group(1) + "}", body)
print("table notes moved outside tabular:", n_notes)
# larger-font, two-row version of the adaptation figure for the single-column page
body = body.replace("width=0.97" + BS + "textwidth]{fig_j_adaptation.png}",
                    "width=0.82" + BS + "textwidth]{fig_j_adaptation_springer.png}")

HEAD = r"""\documentclass[pdflatex,sn-mathphys-num]{sn-jnl}
\usepackage{graphicx}
\usepackage{multirow}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{booktabs}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{url}
\graphicspath{{figures/}{./}}
\newcommand{\sysname}{SignalCheck}
\raggedbottom

\begin{document}
\title[\sysname{} Revisited]{\sysname{} Revisited: Benchmark Artefacts and Modality Scope in Bangla Fake-News Image Detection}

\author*[1]{\fnm{Given} \sur{Name}}\email{author@example.com}
\affil*[1]{\orgdiv{Department}, \orgname{Institution}, \orgaddress{\city{City}, \country{Bangladesh}}}

\abstract{@ABSTRACT@}

\keywords{@KEYWORDS@}

\maketitle
"""

DECL = r"""
\section*{Declarations}
\bmhead{Funding} [TO FILL: funding statement, or ``No funding was received.'']
\bmhead{Competing interests} [TO FILL]
\bmhead{Data availability} The BanFakeMM benchmark is public (doi:10.17632/3r9s7wfbh5.1, CC BY 4.0). [TO FILL: repository URL for code and result files.]
\bmhead{Author contributions} [TO FILL]
\bmhead{Use of AI tools} [TO FILL per publisher policy]
"""

head = HEAD.replace("@ABSTRACT@", abstract).replace("@KEYWORDS@", kw)
out = (head + body + DECL + "\n" + BS + "begin{thebibliography}{00}" + bib
       + BS + "end{thebibliography}\n\n" + BS + "end{document}\n")
io.open(P / "journal_springer.tex", "w", encoding="utf-8").write(out)
print("wrote journal_springer.tex", len(out.splitlines()), "lines")

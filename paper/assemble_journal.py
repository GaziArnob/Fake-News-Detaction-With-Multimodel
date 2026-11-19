"""Assemble the complete journal manuscript from the section files.

Takes the preamble, title block, architecture figure and bibliography from the
conference build, inserts the three new references, and concatenates the five
journal sections in order.

Output: paper/journal_banfakemm.tex
"""
import io
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
P = Path(r"E:\Project\Fake News Detection\paper")
CONF = io.open(P / "iccit2026_6page.tex", encoding="utf-8").read()
OUT = P / "journal_banfakemm.tex"

SECTIONS = ["journal_abstract.tex", "journal_intro_related.tex",
            "journal_methodology_section.tex", "journal_results_section.tex",
            "journal_discussion_conclusion.tex"]

# ---- pieces lifted from the conference build --------------------------------
fig = re.search(
    r"% Spans both columns.*?\\begin\{figure\*\}.*?\\end\{figure\*\}",
    CONF, re.S)
assert fig, "architecture figure not found"
fig_block = fig.group(0)

bib = re.search(r"\\begin\{thebibliography\}\{00\}(.*?)\\end\{thebibliography\}",
                CONF, re.S)
assert bib, "bibliography not found"
bib_body = bib.group(1).strip()
print(f"conference bibitems: {bib_body.count(chr(92) + 'bibitem')}")

NEW_BIB = r"""
\bibitem{shortcut} R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel,
W. Brendel, M. Bethge, and F. A. Wichmann, ``Shortcut learning in deep neural
networks,'' \emph{Nature Machine Intelligence}, vol. 2, pp. 665--673, 2020,
doi: 10.1038/s42256-020-00257-z.

\bibitem{datasetbias} A. Torralba and A. A. Efros, ``Unbiased look at dataset
bias,'' in \emph{Proc. IEEE Conf. Computer Vision and Pattern Recognition
(CVPR)}, 2011, pp. 1521--1528, doi: 10.1109/CVPR.2011.5995347.

\bibitem{nadeau} C. Nadeau and Y. Bengio, ``Inference for the generalization
error,'' \emph{Machine Learning}, vol. 52, no. 3, pp. 239--281, 2003,
doi: 10.1023/A:1024068626366.

\bibitem{verma} Y. Verma, ``What does 99\% accuracy measure? A reproducible
audit of shortcut learning in a widely used fake news corpus,'' 2026,
arXiv:2609.25006.

\bibitem{verite} S.-I. Papadopoulos, C. Koutlis, S. Papadopoulos, and
P. C. Petrantonakis, ``VERITE: A robust benchmark for multimodal misinformation
detection accounting for unimodal bias,'' \emph{Int. J. Multimedia Inf.
Retrieval}, vol. 13, no. 1, art. 4, 2024, doi: 10.1007/s13735-023-00312-6.

\bibitem{linmodality} H. Lin, H. Liu, S. Cao, J. Li, H. Li, and W. Wang,
``Unveiling modality bias: Automated sample-specific analysis for multimodal
misinformation benchmarks,'' 2025, arXiv:2511.05883.
"""

PREAMBLE = r"""%==============================================================
% BanFakeMM journal manuscript -- assembled build.
%
% Assembled from, and kept in sync with, these section files:
%   journal_abstract.tex
%   journal_intro_related.tex
%   journal_methodology_section.tex
%   journal_results_section.tex
%   journal_discussion_conclusion.tex
% Edit those and re-assemble, or edit here and keep them aligned; do not let
% the two drift.
%
% CLASS. \documentclass[journal]{IEEEtran} is a general-purpose choice. For
% IEEE Access swap in ieeeaccess.cls; for an Elsevier title, elsarticle. The
% body does not depend on which.
%
% ANONYMITY. \blindsubmissiontrue blanks the author block. Set it false and
% fill in the \else branch for a single-blind or camera-ready submission.
%
% FIGURES. fig1_system_architecture.png must sit in figures/ or beside this
% file. The other five figures from the conference build are not used here;
% the journal tables carry those numbers.
%
% EVERY NUMBER in this manuscript is traceable to a CSV in the project root.
% The per-section headers below name the file for each group of figures.
%==============================================================
\documentclass[journal]{IEEEtran}
\IEEEoverridecommandlockouts

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{textcomp}
\usepackage{url}
\usepackage[section]{placeins}

% Keep floats near the text that discusses them.
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.75}
\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.65}
\renewcommand{\dbltopfraction}{0.92}
\setcounter{topnumber}{3}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}

\graphicspath{{figures/}{./}}

\newif\ifblindsubmission
\blindsubmissiontrue      % blank author block for blind review

\newcommand{\sysname}{SignalCheck}

\begin{document}

\title{\sysname{} Revisited: Benchmark Artefacts and Modality Scope\\
in Bangla Fake-News Image Detection}

\ifblindsubmission
  \author{}
\else
  \author{Given~Name~Surname,~\IEEEmembership{Member,~IEEE,}
          Given~Name~Surname,
          and~Given~Name~Surname
  \thanks{Author affiliations and contact details.}}
\fi

\maketitle
"""


def strip_marker(text, marker):
    """Drop a leading section command that the assembled file supplies."""
    return text


chunks = [PREAMBLE]
for name in SECTIONS:
    s = io.open(P / name, encoding="utf-8").read().rstrip()
    if name == "journal_abstract.tex":
        chunks.append("\n" + s + "\n")
        continue
    if name == "journal_methodology_section.tex":
        # place the architecture figure right after the opening subsection
        anchor = "\\subsection{Benchmark and Attack Types}"
        assert anchor in s, "methodology anchor missing"
        s = s.replace(anchor, fig_block + "\n\n" + anchor, 1)
    # the bibitem comments at the end of intro+related are now redundant
    if name == "journal_intro_related.tex":
        s = re.split(r"%={10,}\n% BIBITEMS", s)[0].rstrip()
    chunks.append("\n" + s + "\n")

chunks.append("\n%==============================================================\n"
              "\\begin{thebibliography}{00}\n\n"
              + bib_body + "\n" + NEW_BIB
              + "\n\\end{thebibliography}\n\n\\end{document}\n")

text = "\n".join(chunks)
io.open(OUT, "w", encoding="utf-8").write(text)
print(f"wrote {OUT.name}: {len(text.splitlines())} lines, "
      f"{OUT.stat().st_size} bytes")
print(f"  bibitems total : {text.count(chr(92) + 'bibitem')}")
print(f"  tables         : {text.count(chr(92) + 'begin{table}')}")
print(f"  figures        : {text.count('includegraphics')}")

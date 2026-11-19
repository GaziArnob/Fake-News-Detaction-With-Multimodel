# LREV submission pack

Everything in [BRACKETS] is something only you can fill in. Nothing below invents
a name, an email, a funder or a URL.

---

## STOP: read these three before submitting anything

1. **ICECE overlap.** ICECE's rule #6 forbids a paper that is "currently under
   review" from being submitted to a journal at the same time. The journal
   manuscript shares methodology and several tables with the ICECE paper. Wait for
   the ICECE decision, or ask the ICECE chair in writing and keep the reply.
   Measured wording overlap is about 14% (8-word runs; `analysis/overlap_with_conference.py`),
   but the *empirical* overlap is larger than the wording overlap: Table I's ablation
   rows, the fusion comparison and the baselines are inherited. A similarity checker
   sees wording; an editor sees results.
2. **The manuscript says artefacts "are released with the code".** That has to be
   true before you submit. Put `analysis/`, the result CSVs and the screening
   clusters in a public repository (GitHub or Zenodo) and put the URL in the data
   availability statement. Do not release the BanFakeMM images themselves;
   link to the dataset DOI.
3. **AI assistance.** A large share of this manuscript's text was written with an
   AI assistant. Springer Nature has a policy on this. Check its current wording
   and make the disclosure it asks for. I could not retrieve the policy text, so I
   will not quote it. Disclosing is cheap; being found out is not.

---

## COVER LETTER

[DATE]

The Editors
*Language Resources and Evaluation*

**Re: Submission of "SignalCheck Revisited: Benchmark Artefacts and Modality
Scope in Bangla Fake-News Image Detection"**

Dear Editors,

We submit the enclosed manuscript, a full-length paper, for consideration in
*Language Resources and Evaluation*.

**What the paper is.** BanFakeMM, released in July 2026
(doi:10.17632/3r9s7wfbh5.1), is a Bangla multimodal fake-news benchmark with five
adversarial splits. We audit it as a language resource before using it to evaluate
a detector, and we report what the audit changes about the results.

**What we found.**

- In the two splits whose manipulated images come from an image pipeline, a
  classifier given only file metadata (resolution, format, colour mode, with no
  pixel content) separates the classes with 100.00% and 99.35% accuracy. Re-encoding
  every image through one pipeline removes the artefact; the visual model's
  accuracy falls from 95.38% to 94.35%, so the remaining signal is in the content.
- In the other three splits, every one of the 1,000 manipulated images is
  byte-identical to an authentic image. An image-only model cannot do better than
  chance there, and measured AUROC is 0.48 to 0.52. We take this to be a design
  choice, since those attacks alter the text, but it means a single image-model
  score averaged over five splits mixes measurement with noise.
- On a 1,308-image sealed partition, supervised adaptation improves ranking by
  +0.024 to +0.027 AUROC (95% CIs excluding zero, eight seeds).
- Against the feature families of the closest prior Bangla multimodal system,
  evaluated under one protocol, our visual backbone is 6.1 accuracy points ahead.

**Corrections.** Two conclusions we published in an earlier conference paper do
not survive these controls. We state them as corrections in the manuscript:
the claim that the distribution-shift penalty was mostly a calibration effect, and
the claim that supervised adaptation adds nothing.

**Fit with the journal.** The contribution is the evaluation of a language
resource and of the systems built on it, which falls within the journal's remit on
methods for evaluating resources, technologies and applications. We are open about
a limit: our detector reads images and does not process the Bangla text. The
manuscript argues that part of evaluating a multimodal resource is establishing
which of its splits a given modality can decide at all, and it identifies reading
the article text as the natural extension.

**Prior publication.** [CHOOSE ONE AND DELETE THE OTHER]

[A. If ICECE has decided:] A six-page conference version was [rejected by / accepted at /
published in] ICECE 2026 ([citation or status]). The present manuscript extends it
substantially: the metadata artefact and its control, the per-split analysis, the
enlarged sealed partition, the prior-system baseline, the article-level screen,
and the revised adaptation result are all new, and it corrects two of the
conference paper's conclusions. Shared wording is about 14%. We cite the
conference paper in the text.

[B. If you have the chair's written permission:] A six-page conference version is
under review at ICECE 2026, and we have the organisers' agreement that this
extended version may be submitted. [Attach or quote the permission.]

This manuscript is not under consideration anywhere else.

**Availability.** The benchmark is public under CC BY 4.0. Analysis code,
per-image scores, screening clusters and result tables are at [REPOSITORY URL].

**Suggested reviewers** (no conflicts of interest): [NAME, AFFILIATION, EMAIL] ×3
or 4. Choose people from the manuscript's related work who are not collaborators.
Do not leave this blank if the system asks.

Thank you for your consideration.

Sincerely,
[NAME], [AFFILIATION], [EMAIL], [ORCID]
[Co-authors]

---

## DRAFT DECLARATIONS (Springer journals ask for these)

**Data availability statement** *(a Data Availability Statement is required for
original research in this journal)*:
The BanFakeMM benchmark analysed in this study is publicly available at
doi:10.17632/3r9s7wfbh5.1 under CC BY 4.0. The analysis code, screening clusters,
per-image model scores and result tables supporting the findings are available at
[REPOSITORY URL].

**Code availability:** [REPOSITORY URL]

**Funding:** [None / name the funder and grant number]

**Competing interests:** [None declared / describe]

**Ethics approval and consent:** Not applicable. The study uses a public benchmark of
published news images and involves no human or animal participants. *(Confirm this
matches the journal's wording.)*

**Author contributions:** [Per author, CRediT roles]

**Use of AI tools:** [Describe how an AI assistant was used and for what. Follow the
publisher's current policy on where this goes.]

---

## WHAT THE MANUSCRIPT STILL NEEDS BEFORE SUBMISSION

Status key: **seen** = in a source I read; **check** = I could not retrieve the
page, so confirm it yourself.

| Item | Status |
|---|---|
| Hybrid journal; subscription route has no fee; choose it at submission, not open access | **seen** (publisher pages) |
| Full-length papers typically 18 to 25 pages | **seen** (search snippet of the guidelines) |
| Data Availability Statement required | **seen** (same snippet) |
| Springer Nature LaTeX template (`sn-article`) and journal reference style | **check** |
| Abstract word limit (ours is about 290 words) | **check** |
| Peer review type, single or double blind (decides whether the author block stays empty) | **check** |
| Policy on extended conference papers | **check** (Springer's *proceedings* rule is at least 30% new material, a citation, and an explicit statement; it is not stated to apply to this journal) |
| AI-use disclosure wording | **check** |
| Only one figure; a journal paper this long reads better with three or four | mine to add |
| Author block, affiliations, ORCID | yours |
| Public repository | yours |

The Springer guideline pages sit behind a cookie redirect, so I could not read
them. If you open
`https://link.springer.com/journal/10579/submission-guidelines` and paste the text
here (or save it as a PDF), I can reformat the manuscript to it exactly instead of
guessing.

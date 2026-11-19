# Analysis scripts

Every number in the journal manuscript (`paper/journal_banfakemm.tex`) comes
from one of these scripts. They write their results as CSVs to the project
root, which is where the manuscript's per-section headers point.

Run them with the project virtualenv:

```bash
cd "E:/Project/Fake News Detection/analysis"
"../.venv/Scripts/python.exe" <script>.py
```

`common.py` holds the shared loaders. CLIP embeddings, perceptual hashes and
re-encoded images are cached under `../_cache/`, which survives between
sessions — once warm, re-running is cheap.

## Pipeline order

| Script | Produces | Manuscript |
|---|---|---|
| `shortcut_check.py` | *(diagnostic, prints only)* | Sec. IV-C finding |
| `ceiling.py` | `split_ceilings.csv` | Table II, ceiling column |
| `textsplit_eval.py` | `textsplit_results.csv`, `textsplit_scores_*.csv` | Table II, measured |
| `textsplit_proof.py` | `textsplit_proof.csv` | Table II AUROC, Sec. IV-D |
| `artefact_control.py` | `artefact_control.csv` | Table I, artefact rows |
| `nb_corrected.py` | `fusion_vs_cliponly_nb.csv` | Sec. IV-B statistics |
| `big_sealed.py` | `big_sealed_results_v2.csv` | Table III |
| `adaptation_final2.py` | `adaptation_final{,_seeds}.csv` | Table IV |
| `article_screen.py` | `article_screen.csv` | Table IV lower block, Sec. IV-G |
| `article_map_probe.py` | *(diagnostic)* | Sec. III-D-2 mapping |

`shortcut_check.py` and `ceiling.py` are the two that changed the shape of the
paper; run them first if you are reproducing from scratch.

## Checks

| Script | Checks |
|---|---|
| `validate_journal.py` | braces, environments, labels, refs, cites in the assembled `.tex` |
| `check_results_tex.py` | every numeric claim in the Results section against the CSVs |
| `check_abstract.py` | every numeric claim in the Abstract |
| `check_four.py` | all four section files together |
| `inspect_pdf.py` | page count and unresolved refs in a compiled PDF |

## Building the manuscript

```bash
cd ../paper
"../.venv/Scripts/python.exe" assemble_journal.py
"../_cache/tools/tectonic.exe" -X compile journal_banfakemm.tex --outdir .
```

The five section files (`journal_abstract.tex`, `journal_intro_related.tex`,
`journal_methodology_section.tex`, `journal_results_section.tex`,
`journal_discussion_conclusion.tex`) are the source of truth; the assembled
`journal_banfakemm.tex` is generated. Do not edit the assembled file and the
sections independently.

## Protocol constants

Fixed across every script, in `common.py`:

- **Re-encoding**: 384×384, RGB, JPEG quality 90, EXIF dropped
- **Near-duplicate screen**: pHash ≤ 2 **or** dHash ≤ 2 **or** CLIP cosine ≥ 0.98
- **Thresholds** (selected on the 148-image validation partition, never on test):
  raw 0.09 / 0.12, controlled 0.25 / 0.35
- **Seeds**: 42, 101, 202, 303, 404, 505, 606, 707
- **CLIP**: `openai/clip-vit-base-patch32`, `get_image_features` →
  `pooler_output` → L2 normalise, 512-d, PCA to 64
- **Cross-validation**: `StratifiedGroupKFold` over duplicate clusters,
  2 repeats × 5 folds, Nadeau–Bengio correction for fold dependence

## Note on provenance

These scripts were reconstructed after the original working copies were lost
to a temp-directory cleanup. The results CSVs and the `_cache/` embeddings were
never lost, so no measurement was repeated or changed; `nb_corrected.py` and
`textsplit_proof.py` were re-run after reconstruction and reproduce their saved
CSVs exactly.

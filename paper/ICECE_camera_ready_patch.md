# ICECE camera-ready patch (NOT applied to iccit2026_6page.tex)

Use only if ICECE accepts the paper. Four scoped edits, each replacing text of about the
same length so the 6-page limit holds. Everything below is measured in the journal work;
the source file for each number is named. The submitted `.tex` is untouched.

**Why these edits exist.** The conference paper concludes that the shift penalty is
"calibration" and that supervised adaptation "adds nothing". The journal work shows both
statements rest on the uncontrolled files and on a 194-image partition:

| Claim in the conference paper | What the journal work measured | Source |
|---|---|---|
| Threshold selection recovers +6.1 points | +0.5 after re-encoding (CI [-3.6, +4.6], McNemar p = 1.000) | `shift_controlled_results.csv` |
| Adaptation adds nothing (-1.5, CI [-4.6, +1.5]) | +0.7 (original) / +3.8 (re-encoded) on 1,308 sealed images; AUROC +0.0240 / +0.0273, 8 seeds | `big_sealed_results_v2.csv`, `adaptation_final.csv` |
| CI [-4.6, +1.5] | converges to [-5.2, +1.5] with enough resamples | `analysis/check_ci194.py` |

## Edit 1 -- abstract, lines 125-128

REPLACE
```
a validation-selected threshold recovers most of that without
retraining, reaching 91.2 percent and 96.0 percent. Supervised adaptation on
300 target-style images adds nothing measurable ($-1.5$ points, 95 percent CI
$[-4.6, +1.5]$); an earlier draft reported a gain here, which we retract.
```
WITH
```
a validation-selected threshold recovers most of that on the original files
(91.2 and 96.0 percent), but only $+0.5$ points once every image is re-encoded
through one pipeline. Supervised adaptation on 300 target-style images is
indistinguishable from zero in accuracy on this 194-image partition ($-1.5$
points, 95 percent CI $[-5.2, +1.5]$) yet improves AUROC by $+0.024$ on a larger
sealed partition.
```

## Edit 2 -- Section IV-C, lines 547-551 ("Almost the entire shift penalty ...")

REPLACE
```
Almost the entire
shift penalty is therefore calibration rather than failed visual
discrimination: the ranking already separates the classes, and only the
operating point was wrong.
```
WITH
```
On the original files this looks like a calibration
effect, but it is largely an artefact of storage: re-encoding every image
through one pipeline removes the metadata signal, and the same threshold then
recovers only $+0.5$ points.
```

## Edit 3 -- Section IV-C, lines 560-572 (adaptation paragraph)

REPLACE the italic opening sentence
```
\emph{Supervised adaptation adds nothing measurable, for either
configuration.}
```
WITH
```
\emph{Supervised adaptation is not resolved by accuracy on this partition.}
```
and REPLACE the last sentence of the following paragraph
```
We report the measurement under the present protocol---no gain, with an
interval spanning zero.
```
WITH
```
Measured on 1,308 sealed images over eight seeds, the same step improves AUROC by
$+0.024$ (95 percent CI $[+0.017, +0.031]$); the interval above was too wide to
see it.
```

## Edit 4 -- Section V-A and Conclusion

Section V-A, lines 632-640: REPLACE
```
Ours is firmly the second---a threshold of
0.09, chosen without ever seeing the test partition, recovers 91.2 percent
accuracy and 96.0 percent fake-recall---so what failed at the default cut-off
was calibration, ...
```
WITH
```
On the original files the second looks
dominant, but re-encoding shows that much of it came from storage metadata, and
adaptation improves ranking, so we do not claim the shift is purely a matter of
calibration, ...
```
and DELETE the next paragraph ("It also reframes what the 300 adaptation images were
doing ...") except its last two sentences about the boundary of the shift.

Conclusion, lines 686-694: REPLACE
```
Supervised adaptation adds nothing measurable once the threshold is chosen
honestly, for either configuration.
```
WITH
```
Supervised adaptation is invisible in accuracy at a fixed threshold on this
partition but improves AUROC on a larger sealed one.
```
and REPLACE the final paragraph's "our results locate most of it in calibration rather
than in the learned representation, which is both cheaper to fix and easier to get
wrong" WITH "part of it is a property of the benchmark rather than of the model".

## After applying
1. Recompile with Tectonic; confirm exactly 6 pages.
2. Re-read the abstract against Table IV of the paper: Edit 1 now cites two numbers
   (-5.2 and +0.024) that are not in the conference tables, so add one sentence in
   Sec. IV-C pointing to the journal version, or drop those two numbers.
3. The journal manuscript cites the conference paper as the source of the retracted
   claims, so keep the two consistent before either is published.

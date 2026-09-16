# Round 2: targeted verification of reporting repairs

Date: 2026-09-15. Review scope: the two initial-review findings and the separately assigned table-caption corrections in `manuscript.md`. No new experiments or full statistical rerun.

`calibration_status: NOT_CALIBRATED`

`criteria_binding_unavailable`

## Verified repairs

- **Training/evaluation geometry — resolved.** Section 2.2 now explicitly distinguishes 256 × 256 training patches sampled directly from aligned 1024 × 1024 caches from 512 × 512 evaluation inputs. The valid evaluation field is tied to the 512 grid. This agrees with the source behavior checked in round 1.
- **Loss interpretation — resolved.** Section 2.4 now states the B/A/V/X cross-entropy weights, valid noncrossing mask and Hessian-tangent probability-gradient penalty, and entropy-weighted squared logit displacement without a valid-label mask. The original input-ablation versus intact-loss-feature distinction remains. These descriptions agree with the checked implementation.
- **Table-caption scope — resolved in prose.** Table 2 describes five dimensionless structural scores and directs reference errors to accompanying results; Table 3 describes ICC estimates rather than promising an ICC-contrast column. This check verifies caption wording against the main agent's specified table columns, not final rendered table layout.

## Decision and remaining boundaries

The two substantive reporting findings are closed. No additional scientific revision is required by this bounded re-review for delivery as an explicitly labeled initial draft. D2 Methodological Rigor and D5 Writing Quality meet the previously flagged local criteria; the other round-1 judgments remain unchanged within their original coverage. No new contradiction was introduced by these edits.

This is not submission approval. Author-confirmed ethics, contributions, funding, conflicts, access/licensing, and final availability declarations remain necessary before submission. The manuscript retains its substantive limits: historical exploratory cohorts, 224-pixel repeat imagery, separate structural/repeat cohorts, incomplete legacy training provenance, three-seed short retraining, unadjusted intervals, and no independent clinical or cross-camera confirmation. These acknowledged limits do not block initial-draft delivery.

Exact evaluator/writer contract JSON was unavailable. No exact F-condition, integrity, venue-readiness, or contract PASS is asserted. Final body word count, references, source-package completeness, and rendered document/table quality remain with the main agent's delivery checks.

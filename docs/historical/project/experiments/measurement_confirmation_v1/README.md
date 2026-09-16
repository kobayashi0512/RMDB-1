# RMDB readout measurement confirmation: pre-experiment package

**Status:** ready for a new cohort; no prospective image, annotation, model output, or clinical outcome has been read by this package.

This is an experiment package, not a manuscript.  It fixes the cohort requirements, comparison arms, data lock, metrics, and interpretation rules before the next dataset is opened.  It is deliberately designed to test whether different readouts expose a structural-fidelity/repeatability trade-off; it does not assume that a readout "resolves" it.

## What has already been frozen

`historical_result_freeze.json` is the source registry for the already-inspected evidence; its verified, non-overwriting content snapshot is `frozen_historical_evidence_v1/`.  It separates the following quantities, which must never again be interchanged:

| Quantity | Frozen value | Use |
|---|---:|---|
| Fundus-AVSeg reference images | 100 | historical structural reference only |
| mBRSET test images | 982 | full public test partition |
| mBRSET test patients | 258 | full public test partition |
| mBRSET test eyes | 507 | full public test partition |
| mBRSET repeated eyes | 475 | repeatability analysis unit |
| mBRSET patients contributing repeated eyes | 255 | bootstrap cluster count for repeatability |
| mBRSET repeated images | 950 | image count for repeatability |

All historic results remain exploratory.  No number from them may be selected, re-estimated, or merged with the future full confirmation cohort.  In particular, Fundus-AVSeg, mBRSET, RAV, and AFIO are not eligible confirmation data: all have already been inspected by this project; mBRSET is only a 224 px public mirror; RAV lacks public patient/eye/visit linkage; AFIO lacks repeat and camera metadata.  The separately locked HRF IQA runs are high-resolution quality/resolution stress tests, not a substitute for the future cross-camera confirmation, and are now themselves prior-use data.

## Required new cohort

The candidate manifest must pass `cohort_eligibility_audit.py` before a segmentation output is generated.

Minimum design:

- A previously unanalysed cohort, with a per-image `prior_project_use=false` declaration.
- Original, non-upsampled images with long side at least 3000 px.
- The same eye is photographed in one session on at least two identified cameras.  Each camera/configuration has two distinct acquisitions of that eye (four native acquisitions per eye in the complete design).
- Same field centre, dilation state, and acquisition session are recorded.  Camera order is randomized or counterbalanced and all acquisitions are within 30 minutes.
- At least 120 fully qualified eyes from at least 60 participants.  If fewer are available, the output is labelled a precision pilot, never a confirmation.
- A deterministic, manifest-derived annotation subset of 80 eyes.  Every selected image is independently graded for a binary vessel mask by two graders, followed by adjudication.  It is for structural-fidelity estimation only, never model selection.

The two-camera design is intentional: camera/resolution can alter retinal vascular measurements even after images are harmonized to the same resolution.  Cross-camera agreement is therefore reported as a transport/sensitivity result, not silently pooled with within-camera repeatability.

## Locked readouts and contrasts

| Role | Arm | Allowed use |
|---|---|---|
| baseline | `raw_vascx` | VascX output without RMDB readout modification |
| structural readout | `mass` | frozen mass-preserving readout; contextual structural contrast |
| measurement readout | `joint` | frozen occupancy-conditioned VascX identity readout; primary repeatability contrast |
| contextual controls | `smooth`, `cleanup` | fixed non-learned controls; descriptive only |

All weights, preprocessing, output geometry, threshold, and feature code must be recorded in `model_provenance.json` and locked by `freeze_confirmation_manifest.py` before any confirmation output is generated.  No retraining, threshold search, post-processing tuning, feature selection, or clinical-label access is permitted.

## Analysis sequence

1. Complete the candidate manifest using `MANIFEST_SCHEMA.md` and run the eligibility audit.
2. Create `model_provenance.json` from `model_provenance.template.json`; record SHA-256 values of frozen code and weights.
3. Freeze the accepted manifest and provenance with `freeze_confirmation_manifest.py`.
4. Generate one long-format feature table (`image_id, arm, feature, value`) with the locked pipeline.
5. Run `analyze_measurement_gate.py`.  It reports ICC(1,1), within-eye MAD, SEM, and Bland–Altman bias/95% limits of agreement for every arm and configuration.  It additionally reports patient-cluster bootstrap contrasts of `joint` versus `raw_vascx`.
6. Add structural-reference results for the deterministic annotation subset.  Do not alter any result to make a Gate pass.

`analyze_measurement_gate.py` makes no statement of clinical interchangeability.  Without a clinically justified, predeclared absolute tolerance, relative improvement over raw VascX is not proof that a feature is clinically usable.

## Interpretation rule

For the three exploratory-derived confirmation endpoints — `branch_density`, `endpoint_density`, and `occupancy_entropy` — the script uses simultaneous 98.33% patient-cluster bootstrap intervals (Bonferroni over the three endpoints).  Within each camera configuration it labels the `joint` readout as:

- `concordant_relative_improvement` only when the lower CI for ΔICC is above zero **and** the upper CIs for MAD ratio and SEM ratio are below one;
- `mixed_tradeoff` when the three statistics disagree; or
- `not_improved` otherwise.

It always reports LoA/bias and never lets ICC alone determine the label.  Cross-camera results are descriptive sensitivity analyses because a system shift is expected; they do not pass or fail the within-camera Gate.  A conclusion may say that readouts *separate* or *expose* a trade-off only if the locked structural and measurement tables both support that wording.  It may not say that RMDB replaces VascX or resolves device variation.

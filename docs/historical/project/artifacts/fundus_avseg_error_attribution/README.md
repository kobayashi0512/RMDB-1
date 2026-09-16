# Fundus-AVSeg error-attribution audit

> SUPERSEDED (2026-09-05): the 0.5619 evaluation and its oracle analysis below omitted the image preprocessing crop when mapping annotations. The saved image-only transform exactly reproduces all 100 cached RGB images. Corrected evaluation is in `../fundus_coordinate_alignment_audit/summary.json`. The old numbers remain below for audit history, but the bottleneck interpretation must not be used as a scientific conclusion.

This is a post-hoc diagnostic of already locked predictions. It did not fit thresholds, select a checkpoint, read CIMT data, or tune on the labels.

## Overall metrics

| Readout | Vessel Dice | Artery Dice | Vein Dice | Mean A/V Dice | Type accuracy | Brier |
|---|---:|---:|---:|---:|---:|---:|
| VascX | 0.6011 | 0.5513 | 0.5725 | 0.5619 | 0.5148 | 0.0299 |
| Mass readout | 0.6123 | 0.5595 | 0.5792 | 0.5693 | 0.5421 | 0.0304 |
| Final readout | 0.6050 | 0.5543 | 0.5751 | 0.5647 | 0.5235 | 0.0299 |

## What the gap descriptors mean

For each VascX image, `support_gap_descriptor = 1 − vessel Dice` describes the remaining vessel-support error, while `identity_gap_descriptor = vessel Dice − mean A/V Dice` describes the additional loss after restricting evaluation to artery and vein identity. These are not additive causal percentages because support, identity, boundary, crossing, and domain errors interact.

## Disease and quality strata

The complete machine-readable strata are in `summary.json`. The largest fixed strata differences are retained as failure-domain evidence; no subgroup was removed.

Across the 100 images, vessel Dice and mean A/V Dice were strongly aligned (Pearson r = 0.998; descriptive only because both metrics share the same prediction and annotation). Mean A/V Dice was 0.5776 in the fixed high quality stratum and 0.4854 in the low quality stratum; it was 0.5024 in diabetic retinopathy and 0.5734 in normal images. These contrasts identify image quality and disease related appearance as stress domains, not as exclusion criteria.

## Fixed oracle diagnostic

Replacing predicted vessel support with the annotation while retaining frozen predicted identity raises mean A/V Dice to 0.8458 (paired diagnostic gain 0.2839, 95% bootstrap CI 0.2560 to 0.3125). Replacing predicted identity with the annotation while retaining predicted support raises it to 0.6736 (gain 0.1117, 95% bootstrap CI 0.1023 to 0.1209). These counterfactual upper bounds indicate that missed or fragmented vessel support is the larger bottleneck on this set, while identity errors remain material. They are not achievable model results and do not form an additive decomposition.

## Interpretation

The audit is consistent with support/domain error being the dominant observable limitation: vessel Dice is substantially higher than mean A/V Dice, and the low-quality and diabetic-retinopathy strata are lower than normal or high-quality strata. The mass readout provides a small but consistent improvement, while the final routed readout is smaller. This supports an evidence-bound conclusion rather than a claim of a large segmentation gain.

A true causal decomposition would require fixed oracle replacements (ground-truth support with predicted identity, predicted support with ground-truth identity, boundary-tolerant scoring, and crossing exclusion) and should be treated as a diagnostic extension, not a new model-selection loop.

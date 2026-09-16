# Bounded internal review of the initial draft

Review date: 2026-09-15. Title: *RMDB Dual Readouts for Retinal Vascular Measurement*. Scope: complete initial-draft prose, the supplied bibliography metadata, selected source implementations, and the source rows supporting the main historical, mechanism, confidence-control, and residual-extension claims. Tables were being inserted by the main agent during review; their final layout and every cell are outside this review receipt.

`calibration_status: NOT_CALIBRATED`

`criteria_binding_unavailable`

This follows the earlier paper-blind five-dimension plan as a documented fallback. Exact evaluator/writer contract JSON and failure-condition definitions were unavailable. The writer's summarized seven commitments were supplied. No exact contract, F-condition, submission-readiness, or venue-alignment PASS is claimed. This is one internal evaluator, not an independent external review panel.

## Dimension Scores

| Dimension | Workflow judgment | Categorical judgment | Anchored assessment |
|---|---|---|---|
| D1 Originality | pass | MEETS | Introduction paragraphs beginning “Prior work already studies” and “The contribution is a controlled comparison,” plus §4.1, identify a bounded intervention study and acknowledge adjacent retest/toolbox work. No firstness or whole-system superiority claim requires a stronger novelty case. Literature coverage is targeted, not exhaustive. |
| D2 MethodologicalRigor | warn | PARTLY_MEETS | `repairable_method_reporting_gap`: §2.2 and §2.4 need the two implementation clarifications below. Parallel readouts, common extractor, exploratory chronology, final-epoch selection, limited seeds, and statistical estimands are otherwise sufficiently delineated for the stated initial-draft purpose. |
| D3 EvidenceSufficiency | pass | MEETS | Checked headline rows agree with the historical structure/agreement tables, mechanism_pair_estimates.csv, new_method_paired_contrasts.csv, and residual external group contrasts. §§3.1–3.5 preserve adverse findings and separate point estimates from intervals. This is source consistency checking, not rerunning inference or all bootstrap intervals. |
| D4 ArgumentCoherence | pass | MEETS | §§1, 2.6, 3.2 and 4.2 maintain the separation between algebraic accounting and acquisition causes; §§3.4–3.5 distinguish legacy and retrained configurations and relative versus absolute agreement. The conclusion remains appropriately exploratory. |
| D5 WritingQuality | warn | PARTLY_MEETS | `repairable_presentation_gap`: §2.4's named loss terms are too compressed to make the implemented penalty interpretable; the geometry sentence can be read as applying to training. Final word counts, inserted table cross-references, references, and author declarations are controlled by the main agent rather than certified here. |

## Failure Condition Checks

Exact F1/F2/F3/F6/F4/F5/F0 evaluation is NOT_ASSESSED because the contract definitions are unavailable. No source-supported `unsupported_core_novelty`, `invalid_core_inference`, `unsupported_central_claim`, or `broken_question_evidence_conclusion_chain` was identified within the checked coverage. The warnings below require reporting edits, not new experiments or scientific reframing.

## Review Body

### Two substantive reporting repairs

1. **Clarify training versus evaluation geometry.** Anchor: §2.2, “RGB images are resized bilinearly to 512 × 512,” and §2.4, “two 256-pixel crops per training image per epoch.” The former can imply that training crops originate on the 512 grid. In `src/train_controlled_residual_ablation.py`, `Cases.get` reads cached RGB/logits and aligned annotations, while `Crops.__getitem__` samples coordinates 0–768 and extracts aligned 256-pixel patches from the 1024 grid. `feature_tensor` then computes crop-local Hessian features. Evaluation resizes to 512. **Minimum fix:** state that 512 refers to evaluation, and that retraining extracts corresponding RGB/logit/nearest-neighbor-aligned-label patches directly from the 1024 caches before calculating features. This also exposes the scale difference honestly. No numerical results need alteration.

2. **Define the directional and identity penalties, including their masks.** Anchor: §2.4, “The shared retraining objective combines cross-entropy with soft Dice, directional, vessel binary-cross-entropy and identity terms.” `src/train_udct_av_pilot.py:152` implements class-weighted cross-entropy with weights (0.20, 1, 1, 2), ignoring target −100. Its directional penalty sums absolute central-difference A/V probability gradients projected onto the local Hessian tangent, weighted by vesselness × (1 − normalized four-class entropy), valid labels, and exclusion of crossing targets. Its identity penalty is the spatial/channel mean of (1 − entropy) × squared **logit** displacement from the raw logits; unlike directional and supervised terms, it is not masked to valid or non-crossing labels. **Minimum fix:** add a compact definition, including that the identity penalty operates on logits and that intact features supply all loss terms despite input ablation. Avoid describing it as a probability-space KL penalty or applying a common valid-pixel mask to all losses. Existing weights and scope statements are correct.

### Checks supporting the central results

- Historical structure.csv supports Dice 0.8556→0.8647, precision 0.9396→0.9096, thin-centreline recall 0.7971→0.8365, and the stated structural tradeoff. Historical agreement.csv supports the nine ICC point-estimate increases, three endpoints with both favorable ICC and MAD intervals, the adverse A/V response, and numerical preservation of identity entropy.
- mechanism_pair_estimates.csv supports the test squared-log-error change 0.0205898438 [0.0157089446, 0.0259167849], cross term 0.0036672605, reweighting-square term 0.0169225832, and the alternative artery/vein allocations. §2.6's algebra sums correctly and matches `readout_mechanism_experiment.py`; the draft correctly avoids adding both decompositions together.
- The confidence source implements the specified per-image absolute-update budget and asserts a floating-point matching tolerance. The ratio/MAD deterioration and component-count MAD contrast are supported by the mechanism source tables. “Exactly” denotes the analytical construction, not bitwise equality of floating-point outputs.
- Residual external group contrasts support full-minus-legacy Dice +0.0076772197 [0.0045079011, 0.0099955355]; full-minus-raw vessel-mass ICC −0.0217922629 and MAD +0.0006794403; branch-density ICC −0.0800143895 and MAD +0.0001033636; component-count ICC +0.0793456474 and MAD +0.3901754386, with the reported intervals and directions.
- `three_readout_repeatability.py:179` supports the unequal-view ICC and all-view MAD/pooled within-eye SD equations. `analyze_residual_ablation_external.py` confirms paired patient/seed resampling and three-checkpoint metric means, not probability ensembling. Its LoA use the first two views ordered by view and image identifier. The manuscript explicitly distinguishes historical first-two-view SEM/LoA summaries from extension all-view pooled SD; the test cohort has two views per eye, so this distinction does not create a test-cohort estimand discrepancy. The historical SEM construction itself was not independently rebuilt in this review.

### Boundaries and nonissues

The bibliography file contains 11 source records with locators and stated support scopes. This review checked how the manuscript uses those scopes; it did not independently repeat the main agent's primary-source verification. The table-2 caption's promise of six absolute errors was already assigned to the main agent for correction and is not a new finding. Initial-draft pending-author declarations and absence of newly generated figures are consistent with the assigned scope. They are not reasons to demand new experiments or to block delivery of an explicitly labeled initial draft. Complete source tables must remain accessible with the delivered package where the prose says they are available.

## Evaluator Decision

**Minor reporting revision for initial-draft delivery.** Apply the two method clarifications, finish the already assigned table/declaration work, and retain the current bounded conclusions. No central scientific blocker was found within this review's coverage. This decision does not imply publication acceptance or submission readiness and is not an exact evaluator-contract emission.

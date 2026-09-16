# Backbone development screen: recovery amendment, 7 September 2026

This is a prospective execution correction for the next run, not a claim that
the legacy experiments were preregistered or completed. Legacy files remain
untouched in `results/backbone_candidates_v1`. New results are isolated in
`results/backbone_candidates_recovery_20260907`.

## Why a new run is required

Two old baseline logs reached eight epochs, then appended a restart from epoch
one. Their overwritten histories contain three epochs and there is no saved
optimizer or random state. Those runs cannot be exactly resumed. Their partial
scores must not be treated as completed baseline results. No old candidate has
a complete 12 epoch summary.

The RAV cache was anatomically cropped, whereas the previous loader only
resized raw annotations. All 206 cached RGB images have now been reconstructed
pixel for pixel using the recorded image transform (204 cases) or its original
image-only detector (two cases). The identical transform is applied to masks
using nearest-neighbor interpolation. No label or Dice score selects the
transform. The audit is `data/rav_geometry_recovery_20260907/audit.json`.
This correction is applied to EVERY candidate, including the baseline. Any
gain relative to a legacy misaligned run is not a backbone improvement.

## Actual candidate implementations

| Code name | Implemented change to the occupancy branch |
| --- | --- |
| baseline | Existing ResNet34 U-Net with its FIVES checkpoint |
| highres | Compact three-scale U-Net initialized from scratch |
| global_local | Compact U-Net with global channel context gating |
| dualstream | Compact U-Net with RGB and repeated high-pass green channels concatenated |
| direction | Compact U-Net with RGB and Hessian-derived channels concatenated |

These are screening prototypes. In particular, `dualstream` does not yet mean
two independent encoders; `global_local` is not a whole-image/patch exchange
network; `direction` is not a novel directional convolution. All use the same
identity/router architecture, but its parameters are trained jointly with
occupancy. VascX logits and the mass corrector are frozen. Merely renaming a
candidate does not establish algorithmic novelty.

## Fixed execution for this screen

- Five candidates, seeds 20260906, 20260907 and 20260908: 15 jobs.
- At most two jobs concurrently on the local MPS device; seed-first scheduling.
- 12 epochs, batch size 4, learning rate 0.00005, training output 512 pixels,
  evaluation output 768 pixels, crop sizes 384/512/640/768. This is a pilot
  budget, not proof of convergence. The old protocol's batch size 2 was not the
  actual launched value and is superseded here.
- RAV: 169 training images, 30 validation images, seven analog-camera images
  excluded from this screen. FIVES official training: 480 train and 120 val.
- Each seed chooses a split; candidates with the same seed use exactly the
  same split. Thus seeds are repeated development splits, not three training
  initializations on a single fixed split.
- Existing composite checkpoint selection is retained for all candidates:
  RAV mean A/V Dice + 0.25 times FIVES vessel Dice minus 0.10 times RAV log A/V
  ratio error. Report its individual components as well as the composite.
- No CIMT test, Fundus-AVSeg labels, FIVES official test labels, mBRSET outcomes,
  or RAV analog holdout evaluation is used by this queue.

## Interpretation limits and completion report

Compare only same-seed candidates in this corrected queue. Report all seeds,
per-image outputs, mean and dispersion, A/V Dice, vessel Dice, clDice, A/V ratio
error and parameter/runtime cost. Do not infer significance from three runs.
The baseline is warm-started but the compact alternatives are not. Historical
FIVES checkpoint overlap with current development partitions has not been
excluded. The shared mass corrector also has historical RAV development
exposure. These development scores cannot establish an independent test gain
or a pure architecture effect. A winning candidate would need a separately
designed, exposure-audited confirmation; this queue does not trigger one.

The old proposed thin-vessel recall/precision success gate is not implemented
in the current evaluator, and must not be declared passed. No candidate may
be advertised as publication-ready or SCI Q2 on these scores alone. Twelve
epochs may disadvantage a randomly initialized model; a plateau, failure or
negative result must be reported rather than hidden by selective comparison.

## Recovery and supervision

Every completed epoch atomically saves model, optimizer, best checkpoint,
history, configuration, split and Python/NumPy/Torch/MPS RNG state. A partial
epoch is replayed from the last complete checkpoint. Source and geometry-audit
hashes prevent mixing changed code or data in a resumed queue. An explicit
PAUSE marker stops new dispatch before workers are stopped. A failed job is
not silently retried. Resume/pause equivalence is tested separately; QA scores
are not research results. The hourly monitor must respect a user pause and
must not restart the unsafe legacy queue.

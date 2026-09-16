# RMDB–BiRAP V2 final audit

Status: complete with negative results. This audit is development only and did not read the CIMT fixed test or use mBRSET clinical outcomes for tuning.

## Public mBRSET output integrity

The public 224 × 224 mirror contains 1,287 patients, 4,884 images, and 2,359 repeated eyes. All 4,884 VascX logit arrays have shape `(1024, 1024, 4)` and finite values. All 4,884 occupancy arrays have shape `(512, 512)` and finite values. These outputs are exploratory and are not an independent high-resolution clinical validation.

## Patient-grouped repeatability

On the held-out patient-grouped test split (475 repeated eyes), six of the nine prespecified features met ICC(1,1) ≥ 0.60 with the 95% bootstrap lower bound also ≥ 0.60: vessel mass, artery mass, vein mass, vessel density, skeleton density, and branch density. A/V ratio, component count, and identity entropy did not pass. The earlier manuscript statement of four of nine must be reconciled with this current audit before submission.

## Outcome-blind adapter

The paired-view adapter reduced within-eye MAE for all nine features but reduced ICC for eight of nine. Component count improved from 0.4630 to 0.5030 but remained below the threshold. The adapter branch is therefore a negative result and was stopped without retuning.

## V2 residual branches

All four prespecified residual branches were stopped when held-out FIVES vessel Dice degraded while RAV artery/vein Dice remained essentially unchanged:

| Variant | RAV mean A/V Dice | FIVES vessel Dice at stop | Decision |
|---|---:|---:|---|
| residual_joint | 0.8542 | 0.6752 | stop |
| residual_topology | 0.8547 | 0.6703 | stop |
| residual_phenotype | 0.8542 | 0.6551 | stop |
| residual_full | 0.8550 | 0.6458 | stop |

The V2 design therefore does not support a claim of improved cross-domain segmentation or phenotype quality. Do not continue outcome-driven tuning. Report these branches transparently as negative/unstable ablations and retain the frozen baseline for the manuscript.


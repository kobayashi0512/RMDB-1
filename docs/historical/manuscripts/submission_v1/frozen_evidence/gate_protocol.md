# Measurement gate for the submission revision

## Current evidence status

This package is an exploratory, outcome-blind measurement audit. It reports every
one of the 12 phenotypes; it does not create a new clinical feature subset after
viewing the held-out test data. Every row in `table3_repeatability_agreement.csv`
reports ICC(1,1) with a patient-clustered 95% CI, within-eye MAD, SEM, and 95%
Bland-Altman limits of agreement (LoA). ICC is therefore not used as a surrogate
for absolute agreement.

Only branch density, endpoint density, and occupancy entropy show a positive
patient-clustered ICC contrast and a positive MAD-reduction CI in this audit.
The remaining ICC improvements are reported as trade-offs, not promoted as a
measurement improvement.

## Prospective confirmation criterion

Before analysing a new cohort, freeze the readout, phenotype operator, patient
split, and the following dual criterion. A phenotype may be labelled
measurement-admissible only if, on the untouched confirmation set, (1) the lower
bound of its ICC(1,1) 95% CI is at least 0.60 and (2) the upper bound of the
within-eye MAD ratio relative to raw VascX is no greater than 1.00. SEM and LoA
must be reported for both readouts; their clinically tolerable ranges must be set
by the clinical application, not inferred from this public dataset.

No clinical outcome may be used to select a readout, a threshold, or a phenotype.

# RMDB submission v1 frozen evidence package

This directory is the only source of numerical claims for the revised manuscript.
It is generated from the bounded `final_measurement_controls` artifact and contains
no raw patient images.

- `frozen_manifest.json` records hashes and the canonical source scope.
- `table1_cohort_partition.csv` fixes the 978-image development and 982-image test
  partitions; repeatability uses 945 and 950 images, respectively.
- `table2_structural_fidelity.csv` is the structural table.
- `table3_repeatability_agreement.csv` adds within-eye MAD, SEM, and LoA to ICC.
- `gate_protocol.md` separates the present exploratory audit from a future
  pre-registered confirmation gate.
- `external_confirmation_preflight.json` documents why the available AFIO material
  is not yet a cross-camera repeated-image confirmation cohort.

Historical outputs remain preserved in their original artifact directories for
traceability. They are not deleted and are not allowed to supply manuscript numbers
unless re-derived into this package.

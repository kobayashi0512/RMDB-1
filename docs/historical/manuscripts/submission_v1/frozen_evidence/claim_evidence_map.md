# Claim-to-evidence map

| Submission claim | Canonical evidence | Permitted wording | Prohibited wording |
|---|---|---|---|
| Mass preservation changes structural performance | Table 2, 100 previously inspected Fundus-AVSeg images | **separates a sensitivity-precision trade-off** | resolves or universally improves segmentation |
| Occupancy conditioning changes repeatability | Table 3, patient-separated mBRSET test split, 475 repeat eyes and 950 images | exposes that ICC and absolute agreement can disagree | all 9 ICC gains are measurement improvements |
| RMDB-BiRAP adds value over VascX | Same frozen VascX outputs are the raw comparator in Tables 2-3 | evaluates which VascX-derived readout is defensible for a stated measurement claim | replaces VascX or establishes a better foundation segmentation model |
| Clinical utility | No reconciled canonical clinical result | no clinical increment claim in this revision | hypertension prediction, transportability, or risk stratification |
| Cross-camera translation | `external_confirmation_preflight.json` | required next confirmation step | already demonstrated |

The relevant prior work is VascX (a segmentation and feature-consistency pipeline),
retest-reliability work showing that segmentation quality does not determine every
biomarker's repeatability, and the recent evidence that camera/resolution changes
alter retinal vascular measurements. RMDB-BiRAP is positioned downstream of VascX:
it assesses the measurement suitability of alternative readouts from a frozen
VascX output, rather than claiming to replace the VascX model.

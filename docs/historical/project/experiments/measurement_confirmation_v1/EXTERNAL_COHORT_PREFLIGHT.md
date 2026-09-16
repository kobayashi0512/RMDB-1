# External cohort preflight (2026-09-09)

No public dataset found in this check satisfies every confirmation criterion.  These are not interchangeable substitutes.

| Candidate | What it offers | Why it is not the confirmation cohort |
|---|---|---|
| Cunnac et al. 2026 camera/resolution study | Same subjects photographed on Topcon TRC NW6S and Canon CR-2; native resolutions up to 5184×3456; useful design precedent. | 54 eyes from 27 participants, below the 120-eye/60-participant requirement; public raw-data access not established here; its published design has one image per eye/camera rather than two within-camera acquisitions. It could only support a small device-sensitivity pilot if access is granted. |
| FIRE registration dataset | Public 134 image pairs from 39 patients, acquired on a Nidek AFC-210 at 2912×2912. | One camera and 2912-px resolution; registration pairs are not paired cross-camera repeats. |
| `smartretina2025/paired_retina` dataset card | Claims 327 simultaneous tabletop/portable patient pairs. | The current card states that the full data will be released after acceptance and lists a dataset size under 1 KB; raw images, original resolution, acquisition metadata, and repeat captures are unavailable. It must not be cited as obtained data. |
| Local Fundus-AVSeg, mBRSET, RAV, AFIO | Already inventoried in `local_candidate_inventory_20260909.json`. | All have prior project exposure and each fails at least one high-resolution, repeat, camera, or linkage condition. |

## Acquisition request to a collaborating clinic or data custodian

Ask for a metadata-only feasibility extract before any image transfer.  It must contain the fields in `MANIFEST_SCHEMA.md` for a proposed cohort and confirm that images have not previously been shared with or analysed by this project.  The requested capture protocol is:

1. For each participating eye, two native captures with Camera A and two with Camera B, ideally with distinct vendors/models and native long side ≥3000 px.
2. Same field centre and dilation state; camera order randomized/counterbalanced; all four captures within 30 minutes.
3. De-identified patient, eye, session, acquisition, camera-model, timestamp, native resolution, and checksum fields retained.
4. At least 120 eligible eye-sessions from 60 participants; include a feasibility buffer for image-quality exclusions.
5. Permission for a deterministic 80-eye subset to receive two independent binary-vessel annotations and adjudication.

Only after the metadata extract passes `cohort_eligibility_audit.py` should the team request raw images or run VascX/RMDB inference.

## Sources

- Cunnac et al. demonstrated camera and resolution sensitivity using paired acquisitions from 27 participants: [Scientific Reports study](https://www.nature.com/articles/s41598-026-44196-y).
- FIRE dataset documentation: [FIRE project page](https://projects.ics.forth.gr/cvrl/fire/).
- The unreleased paired-retina card: [Hugging Face dataset card](https://huggingface.co/datasets/smartretina2025/paired_retina/blob/main/README.md).


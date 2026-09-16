# Candidate-cohort manifest schema

One row is one original image.  Use UTF-8 CSV and do not replace unknown values with fabricated identifiers.

| Column | Required | Meaning |
|---|---|---|
| `image_id` | yes | Immutable unique image identifier. |
| `patient_id` | yes | De-identified participant identifier; supports clustered bootstrap and leakage checks. |
| `eye` | yes | `left` or `right`. |
| `session_id` | yes | Visit/acquisition session shared only by same-session images. |
| `acquisition_id` | yes | Unique capture identifier; two values are required per eye/camera configuration. |
| `camera_id` | yes | Manufacturer + model, not a generic vendor name. |
| `original_path` | yes | Path to the native image, before project preprocessing/resizing. |
| `native_width` / `native_height` | yes | Pixels in that original file. |
| `capture_time` | yes | ISO-8601 time; used to verify the 30-minute session window. |
| `field_center` | yes | e.g. `disc`, `macula`, or a protocol-defined code. |
| `dilation_state` | yes | `dilated`, `undilated`, or protocol-defined state. |
| `prior_project_use` | yes | Must be `false` for every confirmation image. |
| `sha256` | yes | SHA-256 of the original image file. |
| `quality_status` | yes | `accepted` or a prospectively defined exclusion code. |

The eligibility script also accepts optional `annotation_selected` and `annotation_status`.  They are populated only after the locked manifest is created.  A manifest is ineligible if an ID, patient/eye/session key, camera model, native image dimensions, capture time, checksum, or prior-use declaration is missing.

Do not put image-level model scores, phenotype values, segmentations, labels, or clinical outcomes into this manifest.


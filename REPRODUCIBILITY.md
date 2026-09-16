# Reproduction status

Status: **partial artifact archive, not a verified end-to-end reproduction package**.

The original scripts are copied byte-for-byte, including historical code and absolute paths. `requirements.txt` is the original broad dependency specification, not a tested lockfile. Some scripts additionally import OpenCV, SciPy, fundus-preprocessing packages and model packages. Raw data, image-level measurements, weights and caches are deliberately not included. Full reruns require authorized access plus path and environment configuration.

## Result mapping

| Paper evidence | Aggregate files | Original analysis entry points |
| --- | --- | --- |
| Historical structure and agreement | `results/historical/*.csv` | `src/final_measurement_controls.py`, `src/three_readout_repeatability.py` |
| Readout mixtures and confidence controls | `results/mechanism/*.csv` | `src/readout_mechanism_experiment.py`, `src/readout_confidence_experiment.py` |
| Paired A/V decomposition | `results/mechanism/mechanism_pair_estimates.csv` | `src/analyze_readout_mechanism_pairs.py` |
| Three-seed residual extension | `results/residual_extension/*.csv` | `src/train_controlled_residual_ablation.py`, `src/evaluate_residual_ablation_external.py`, `src/analyze_residual_ablation_external.py` |

These entry points are a traceability map, not tested portable commands. Do not run historical queue or training scripts without checking their paths and output locations. No full-run timing or numerical reproduction tolerance has been established for this release.

## Validation performed for upload

`python3 tools/verify_release.py` checks content hashes, parses Python sources without executing them, rejects individual-level CSV field names and checks the frozen cohort counts and historical vessel-Dice row. Passing these checks only establishes archive consistency. It does not reconstruct statistical confidence intervals from restricted records.

The current test partition is 982 images and 258 contributors; the repeated-eye analysis uses 950 images, 475 eyes and 255 contributors. Historical plans with different language or older numbers are preserved under `docs/historical/` for provenance only.

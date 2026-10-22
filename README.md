# RMDB retinal vascular measurement

Research archive for **RMDB Dual Readouts for Retinal Vascular Measurement**, revision of 16 September 2026.

Repository: https://github.com/kobayashi0512/RMDB-1

This repository is publicly visible. Public visibility does not change the project’s data-access boundaries or grant reuse rights. It contains the current manuscript text, analysis code, aggregate experimental results and historical project documentation. It is not a complete backup of the 76 GB local experimental workspace or a self-contained reproduction environment.

## Current manuscript and evidence

- [Complete manuscript text with all five tables, figure captions, declarations and references](paper/RMDB_1_complete_text.md)
- [Manuscript source](paper/manuscript.md), [table source](paper/tables.json), [declarations](paper/endmatter.md)
- [Current aggregate evidence](results/)
- [Figure–manuscript consistency review](paper/图文一致性核查.md)
- [Release scope and data restrictions](DATA_ACCESS.md)
- [Reproduction status and command map](REPRODUCIBILITY.md)

The study compares residual correction and occupancy conditioning separately against raw frozen VascX probabilities under one phenotype extractor. It exposes endpoint-dependent structural-fidelity and same-eye-agreement tradeoffs. It does not establish a universally superior readout or cross-camera clinical interchangeability.

`results/historical`, `results/mechanism` and `results/residual_extension` are distinct analyses. Do not merge historical and retrained estimates or treat previously inspected cohorts as independent confirmation. `docs/historical/` preserves earlier plans, manuscripts and reports, including superseded claims and numbers; it does **not** override `paper/` and `results/`.

## Layout

| Directory | Contents |
| --- | --- |
| `paper/` | Current manuscript text, tables, captions, reference metadata |
| `results/` | Aggregate estimates and contrasts, no participant-level records |
| `src/` | Original project scripts, including legacy work outside the current paper |
| `configs/` | Original configurations and historical development protocols |
| `experiments/` | Confirmation-design and audit scripts; not evidence that confirmation occurred |
| `docs/historical/` | Earlier narrative records, explicitly non-authoritative |
| `SOURCE_MANIFEST.json` | Local source-to-repository mapping and exact copy hashes |
| `SHA256SUMS` | Release integrity checks |

## Quick validation

```sh
python3 tools/verify_release.py
```

This standard-library check validates file hashes, Python syntax, aggregate table schemas and selected frozen counts. It does not rerun segmentation, training, bootstrap estimation or validate clinical claims. See the reproduction limitations before running any legacy scripts.

## Rights and citation

Public visibility does not grant an open-source or dataset licence. No project-wide code licence is provided; source code and third-party materials remain subject to their respective rights and terms. Cite the eventual approved manuscript; no DOI or publication acceptance is claimed. The separate `kobayashi0512/RMDB` repository contains its own analysis history and release assets.

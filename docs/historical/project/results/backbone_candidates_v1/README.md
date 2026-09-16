# Backbone candidate queue

This directory is the outcome blind development namespace for the RMDB BiRAP
V2 backbone comparison. The queue is owned by
`src/run_backbone_candidates_parallel.py` and its state is recorded in
`queue_state.json`. Each candidate and seed has its own log, split, checkpoint,
history, and validation CSVs.

The queue does not read Fundus AVSeg, CIMT, or mBRSET outcomes. It uses only
FIVES train occupancy labels and RAV development A/V labels. A candidate is
not selected from a single seed; the protocol requires all three seeds to be
complete before aggregate comparison.

The queue can be audited without touching workers:

```bash
.venv/bin/python src/monitor_backbone_candidates.py
```

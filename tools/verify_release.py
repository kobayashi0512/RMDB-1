"""Non-executing archive smoke test. Does not rerun scientific experiments."""
from pathlib import Path
import ast
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
count = 0
for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
    digest, name = line.split('  ', 1)
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    count += 1
for p in ROOT.rglob('*.py'):
    ast.parse(p.read_text(), filename=str(p.relative_to(ROOT)))
for entry in json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text()):
    p = ROOT / entry['destination']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == entry['sha256'], str(p)
csv_count = 0
for p in (ROOT / 'results').rglob('*.csv'):
    with p.open(newline='') as f:
        reader = csv.DictReader(f)
        assert not set(reader.fieldnames) & {'id','image_id','patient_id','subject_id','eye_id','path','file'}, str(p)
        assert list(reader), str(p)
    csv_count += 1
with (ROOT / 'results/historical/cohort_partition.csv').open() as f:
    test = next(r for r in csv.DictReader(f) if r['split'] == 'test')
assert [int(test[k]) for k in ['repeat_images','repeat_eyes','repeat_patients']] == [950,475,255]
with (ROOT / 'results/historical/structure.csv').open() as f:
    dice = next(r for r in csv.DictReader(f) if r['endpoint'] == 'Vessel Dice')
assert float(dice['raw_vascx']) == .8556 and float(dice['mass_readout']) == .8647
print(f'PASS: {count} file hashes, Python syntax, {csv_count} aggregate tables, frozen count and Dice checks. No full experimental rerun.')

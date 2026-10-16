#!/usr/bin/env python3
"""Build a genuine SIMG* graph cache from a CSV; report every failed molecule."""
import argparse
import csv
from pathlib import Path
import torch
from toxfm.features import SIMGFeaturizer, canonical_smiles
from toxfm.io import write_json
from toxfm.constants import ENDPOINT_FILES

p = argparse.ArgumentParser(description=__doc__)
source = p.add_mutually_exclusive_group(required=True)
source.add_argument('--input-csv', type=Path)
source.add_argument('--data-root', type=Path, help='Read all five endpoint CSVs in this directory')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--simg-root', type=Path)
p.add_argument('--conformer-seed', type=int, default=42)
p.add_argument('--allow-failures', action='store_true', help='Save partial cache and explicit failure audit')
a = p.parse_args()
audit_path = a.output.with_suffix('.audit.json')
if a.output.exists() or audit_path.exists():
    p.error('Output or audit file already exists')
paths = [a.input_csv] if a.input_csv else [a.data_root / name for name in ENDPOINT_FILES.values()]
rows = []
for path in paths:
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'SMILES' not in reader.fieldnames:
            p.error('{} requires a SMILES column'.format(path))
        rows.extend(reader)
if not rows:
    p.error('Input CSV is empty')
featurize = SIMGFeaturizer(a.simg_root, seed=a.conformer_seed)
graphs, provenance, failures = {}, {}, []
for i, row in enumerate(rows):
    try:
        canonical = canonical_smiles(row['SMILES'])
        if canonical not in graphs:
            graph, meta = featurize(canonical)
            graphs[canonical], provenance[canonical] = graph, meta
    except (ValueError, RuntimeError) as exc:
        failures.append({'row_index': i, 'error': str(exc)})
write_json(audit_path, {'input_rows': len(rows), 'unique_graphs': len(graphs), 'failures': failures})
if failures and not a.allow_failures:
    p.exit(2, 'Graph generation failed; audit saved and no partial cache written. See --allow-failures.\n')
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.save({'graphs': graphs, 'provenance': provenance}, a.output)
print('Saved {} unique SIMG* graphs to {}'.format(len(graphs), a.output))

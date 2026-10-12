"""Audit existing PKU arrays and split membership without evaluating predictions."""
import csv
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = Path(r'D:\AQA\online_skeleton_action_segmentation')
UP = Path(r'D:\AQA\research\keraal_literature\upstream\dest')
DATA = OLD / 'official_data/dataset/PKU-subject'
OUT = ROOT / 'reports'
OUT.mkdir(parents=True, exist_ok=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

result = {'data_root': str(DATA), 'splits': {}, 'split_files': {}, 'overlap': {}}
namesets, bases, subjects, file_records = {}, {}, {}, []
for part in ['train', 'val', 'test']:
    split = UP / 'csv/PKU-subject' / f'{part}1.csv'
    rows = list(csv.DictReader(split.open()))
    names = [Path(row['feature']).name for row in rows]
    assert len(names) == len(set(names)), f'Duplicate in {part}'
    namesets[part] = set(names)
    bases[part] = {name.split('-')[0] for name in names}
    subjects[part] = sorted({int(name.split('N')[1].split('-')[0]) for name in names})
    counts = Counter()
    lengths = []
    missing, nonfinite, mismatch = [], [], []
    label_dtypes = Counter()
    for name in names:
        paths = [DATA / f / name for f in ['features', 'gt_arr']]
        if not all(p.exists() for p in paths):
            missing.append(name)
            continue
        x, y = [np.load(p, allow_pickle=False) for p in paths]
        lengths.append(len(y))
        if x.shape != (len(y), 150) or y.ndim != 1:
            mismatch.append(name)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            nonfinite.append(name)
        label_dtypes[str(y.dtype)] += 1
        assert np.isfinite(y).all() and (y == np.floor(y)).all() and ((0 <= y) & (y < 52)).all(), name
        labels, freq = np.unique(y, return_counts=True)
        counts.update({int(a): int(b) for a, b in zip(labels, freq)})
        for path in paths:
            file_records.append(dict(split=part, name=name, kind=path.parent.name, bytes=path.stat().st_size, sha256=sha(path)))
    result['split_files'][part] = dict(path=str(split), sha256=sha(split), names=names)
    result['splits'][part] = dict(sequences=len(names), frames=sum(lengths), min_frames=min(lengths), max_frames=max(lengths), subjects=subjects[part], base_recordings=len(bases[part]), label_frame_counts=dict(sorted(counts.items())), label_dtypes=dict(label_dtypes), missing=missing, nonfinite=nonfinite, shape_mismatch=mismatch)
for a,b in combinations(namesets,2):
    result['overlap'][f'{a}_{b}'] = dict(sequence_names=sorted(namesets[a]&namesets[b]), base_recordings=sorted(bases[a]&bases[b]), subjects=sorted(set(subjects[a])&set(subjects[b])))
mapping = DATA / 'mapping.txt'
result['mapping'] = dict(path=str(mapping), sha256=sha(mapping), lines=mapping.read_text().splitlines())
config = json.loads((OLD / 'runs/dest_pair_short20_seed17/config.json').read_text())
result['prior_run_matches_upstream_lists'] = {p: set(config[p]) == namesets[p] for p in ['train','val']}
(OUT / 'data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
(OUT / 'data_manifest.json').write_text(json.dumps(file_records, indent=2), encoding='utf-8')
print(json.dumps({'splits': {k: {f:v[f] for f in ['sequences','frames','subjects','base_recordings','missing','nonfinite','shape_mismatch']} for k,v in result['splits'].items()}, 'overlap_counts': {k:{f:len(values) for f,values in v.items()} for k,v in result['overlap'].items()}, 'prior_run_matches':result['prior_run_matches_upstream_lists']}, indent=2))

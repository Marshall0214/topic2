"""Read-only source/split audit and untrained forward causality probe.

Run from project root using dest_gpu Python. Does not train or load weights.
"""
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'sources/specscalpel/upstream'
OUT = ROOT / 'reports/specscalpel_audit'


def main():
    manifest = json.loads((SOURCE.parent / 'manifest.json').read_text())
    mismatches = [entry['path'] for entry in manifest['files']
                  if hashlib.sha256((SOURCE / entry['path']).read_bytes()).hexdigest()
                  != entry['sha256']]
    splits = {}
    for name in ('train', 'val', 'test'):
        with (SOURCE / f'csv/PKU-subject/{name}1.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        splits[name] = {Path(row['feature']).name for row in rows}
    local = json.loads((ROOT / 'configs/dest_compat.json').read_text())['splits']
    result = {'commit': manifest['commit'], 'source_hash_mismatches': mismatches,
              'counts': {k: len(v) for k, v in splits.items()},
              'val_test_overlap': len(splits['val'] & splits['test']),
              'val_subset_test': splits['val'] <= splits['test'],
              'train_equals_local_train_plus_val': splits['train'] == set(local['train'] + local['val']),
              'test_equals_local_test': splits['test'] == set(local['test'])}
    import torch
    torch.manual_seed(17)
    sys.path.insert(0, str(SOURCE))
    from libs.models.SpecScalpel import Model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model(in_channel=12, n_features=64, n_classes=52, n_stages=4,
                  n_layers=10, n_refine_layers=10, n_stages_asb=2,
                  n_stages_brb=3, SFI_layer=list(range(1, 10)),
                  dataset='PKU-subject').to(device).eval()
    x = torch.randn(1, 12, 25, 160, device=device)
    mask = torch.ones(1, 1, 160, dtype=torch.bool, device=device)
    graph = torch.eye(25, device=device)
    changed = x.clone()
    changed[..., 80:] = torch.randn_like(changed[..., 80:])
    with torch.no_grad():
        output = model(x, mask, graph)
        second = model(changed, mask, graph)
    result['model_probe'] = {
        'scope': 'Untrained, synthetic input and identity graph; no accuracy or training claim',
        'device': str(device), 'parameters': sum(p.numel() for p in model.parameters()),
        'shapes': [list(y.shape) for y in output],
        'finite': all(torch.isfinite(y).all().item() for y in output),
        'future_change_past_logits_max_difference': (output[0][..., :80] - second[0][..., :80]).abs().max().item()}
    spec = importlib.util.spec_from_file_location('spec_metric', SOURCE / 'libs/metric.py')
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    import numpy as np
    result['metric_probe'] = {
        'mapping': {'0': 'NONE', '1': 'action'},
        'input': [0, 0, 1, 1],
        'segments': metric.get_segments(np.array([0, 0, 1, 1]), {0: 'NONE', 1: 'action'}),
        'interpretation': 'NONE included; final end index is 3 for a 4-frame sequence'}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit_results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

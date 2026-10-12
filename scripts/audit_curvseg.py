"""Read-only source audit and synthetic probes; no training or downloads."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / 'sources/curvseg/upstream'
OUT = ROOT / 'reports/curvseg_audit/probes.json'


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    torch.manual_seed(17)
    torch.set_num_threads(1)
    result = {'torch': torch.__version__, 'device': 'cpu', 'seed': 17}
    manifest = json.loads((ROOT / 'sources/curvseg/manifest.json').read_text())
    result['commit'] = manifest['commit']
    result['source_files'] = len(manifest['files'])
    result['hash_mismatches'] = [f['path'] for f in manifest['files'] if
        hashlib.sha256((UP / f['path']).read_bytes()).hexdigest() != f['sha256']]
    tree = json.loads((ROOT / 'reports/curvseg_audit/remote_tree.json').read_text())
    remote = {f['path'] for f in tree['tree']}
    paths = ['prompt/text_prompt.py', 'prompt/tools.py', 'libs/loss_fn/circle.py',
             'libs/loss_fn/rdrop.py', 'libs/perturb.py', 'libs/models/SP.py', 'libs/swap.py']
    result['required_sources_present_in_remote'] = {p: p in remote for p in paths}
    protocol = json.loads((ROOT / 'configs/dest_compat.json').read_text())['splits']
    result['splits'] = {}
    for split in ['train', 'val', 'test']:
        with (UP / f'csv/PKU-subject/{split}1.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        names = {Path(r['feature']).name for r in rows}
        ours = set(protocol[split])
        result['splits'][split] = {'rows': len(rows), 'unique': len(names),
            'matches_current': names == ours, 'only_curvseg': sorted(names - ours),
            'only_current': sorted(ours - names)}
    source = (UP / 'train.py').read_text(encoding='utf-8')
    tree_ast = ast.parse(source)
    fn = next(n for n in tree_ast.body if isinstance(n, ast.FunctionDef) and n.name == 'get_arguments')
    ns = {'argparse': argparse}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'upstream/train.py', 'exec'), ns)
    try:
        ns['get_arguments']()
        result['argument_parser'] = 'unexpected success'
    except Exception as exc:
        result['argument_parser'] = f'{type(exc).__name__}: {exc}'
    curve = load_file('curvseg_curve_probe', UP / 'libs/loss_fn/curve.py')
    cross = load_file('curvseg_cross_probe', UP / 'libs/loss_fn/cross.py')
    feature = torch.randn(2, 8, 80, requires_grad=True)
    generator = curve.CurvatureLoss(q=10)
    _, prior, _ = generator.curvature_estimation(feature, device='cpu')
    logits = torch.randn(2, 1, 80, requires_grad=True)
    mask = torch.ones_like(logits, dtype=torch.bool)
    gt = torch.zeros_like(logits)
    loss = cross.TSConstraintLoss(alpha=4, beta=2, loss_type='mse')(prior.unsqueeze(1), logits, mask, gt)
    loss.backward()
    result['curvature_gradient'] = {'input_requires_grad': feature.requires_grad,
        'prior_requires_grad': prior.requires_grad, 'feature_grad_is_none': feature.grad is None,
        'boundary_grad_norm': logits.grad.norm().item(), 'loss': loss.item()}
    result['default_zero_weight_loss'] = cross.TSConstraintLoss(alpha=0, beta=0)(
        prior.unsqueeze(1), logits.detach(), mask, gt).item()
    changed = feature.detach().clone()
    changed[:, :, 40:] += 5 * torch.randn_like(changed[:, :, 40:])
    _, prior_changed, _ = generator.curvature_estimation(changed, device='cpu')
    result['curvature_future_probe'] = {'change_starts_at': 40,
        'past_max_abs_change': (prior_changed[:, :40] - prior[:, :40]).abs().max().item()}
    # Extract the exact self-contained attention class, avoiding unrelated imports.
    model_source = (UP / 'libs/models/pkusub_LaSA_multiscale_fwag.py').read_text(encoding='utf-8')
    model_ast = ast.parse(model_source)
    attention = next(n for n in model_ast.body if isinstance(n, ast.ClassDef) and n.name == 'Linear_Attention')
    ns = {'torch': torch, 'nn': torch.nn, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=[attention], type_ignores=[]), 'upstream/attention', 'exec'), ns)
    # Signature is recorded even if a probe cannot execute in this environment.
    import inspect
    result['attention_forward_signature'] = str(inspect.signature(ns['Linear_Attention'].forward))
    layer = ns['Linear_Attention'](8, 8, 8, n_heads=2, drop_out=0).eval()
    x = torch.randn(1, 80, 8)
    y = x.clone()
    y[:, 40:] += 5 * torch.randn_like(y[:, 40:])
    valid = torch.ones(1, 1, 80)
    with torch.no_grad():
        before = layer(x, x, x, valid)
        after = layer(y, y, y, valid)
    result['attention_future_probe'] = {'change_starts_at': 40,
        'past_max_abs_change': (before[:, :40] - after[:, :40]).abs().max().item()}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

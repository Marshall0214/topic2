"""Validation-only graph decoding interventions; never train or select on test."""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline import Meter, build_model, load_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default='protas_pku_dev_seed17')
    parser.add_argument('--output', default='reports/protas_graph_audit')
    args = parser.parse_args()
    run = ROOT / args.run if Path(args.run).is_absolute() else ROOT / 'runs' / args.run
    cfg = json.loads((run / 'config.json').read_text())
    if cfg.get('normalization') != 'none':
        raise ValueError('This audit only supports the existing unnormalized run')
    if hashlib.sha256(Path(cfg['split_path']).read_bytes()).hexdigest() != cfg['split_sha256']:
        raise ValueError('Split changed')
    names = cfg['selected']['val']
    split = json.loads(Path(cfg['split_path']).read_text())['splits']
    assert names and not set(names) & (set(cfg['selected']['train']) | set(split['test']))
    for relative in ['src/baseline.py', 'sources/protas/upstream/model.py']:
        source_key = str(Path(relative))
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == cfg['source_hashes'][source_key]
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (run / 'train_graph.pkl').open('rb') as handle:
        initial = pickle.load(handle)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cache = {name: load_sequence(name, Path(cfg['data_root'])) for name in names}
    result = dict(run=args.run, split='validation_only', sequences=len(names),
                  interventions='Inference only on existing weights, not retrained ablations', checkpoints={})
    for tag in ['best', 'last']:
        path = run / 'protas' / f'{tag}.pt'
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model = build_model('protas', run / 'train_graph.pkl', cfg['stages'], cfg['layers'], cfg['width']).to(device).eval()
        model.load_state_dict(ckpt['model'])
        graph = model.graph_learner
        a, b = graph.matrix_pre.detach(), graph.matrix_suc.detach()
        ai = torch.as_tensor(initial['matrix_pre'], device=device)
        bi = torch.as_tensor(initial['matrix_suc'], device=device)
        az, bz = a.clone(), b.clone()
        az[0, :] = bz[0, :] = 0
        az[:, 0] = bz[:, 0] = 0
        variants = ['raw', 'learned', 'initial_graph', 'learned_nonnegative',
                    'learned_background_isolated', 'learned_current_progress']
        meters = {key: Meter() for key in variants}
        counts = {key: dict(background=0, changed=0) for key in variants}
        correction = dict(background_sum=0., foreground_sum=0., frames=0)
        max_parity = 0.
        with torch.inference_mode():
            for features, y in cache.values():
                x = torch.from_numpy(features)[None].to(device)
                out, p = model(x, torch.ones(1, 1, len(y), device=device))
                raw = out[-1]
                current = p[-1].clamp(0, 1)
                cumulative = current.cummax(dim=-1).values
                def adjust(pre, suc, status):
                    penalty = torch.einsum('bkt,kK->bKt', 1-status, pre)
                    penalty += torch.einsum('bkt,kK->bKt', status, suc)
                    return raw - graph.eta * penalty
                learned = adjust(a, b, cumulative)
                max_parity = max(max_parity, float((learned-graph.inference(raw, p[-1])).abs().max()))
                logits = dict(raw=raw, learned=learned, initial_graph=adjust(ai, bi, cumulative),
                              learned_nonnegative=adjust(a.clamp_min(0), b.clamp_min(0), cumulative),
                              learned_background_isolated=adjust(az, bz, cumulative),
                              learned_current_progress=adjust(a, b, current))
                raw_pred = raw.argmax(1)[0].cpu().numpy()
                for key, value in logits.items():
                    pred = value.argmax(1)[0].cpu().numpy()
                    meters[key].update(pred, y)
                    counts[key]['background'] += int((pred == 0).sum())
                    counts[key]['changed'] += int((pred != raw_pred).sum())
                delta = learned-raw
                correction['background_sum'] += float(delta[:, 0].sum())
                correction['foreground_sum'] += float(delta[:, 1:].mean(1).sum())
                correction['frames'] += len(y)
        def stats(value):
            return dict(min=float(value.min()), max=float(value.max()),
                        negative_fraction=float((value < 0).float().mean()),
                        background_column_mean=float(value[:, 0].mean()),
                        foreground_columns_mean=float(value[:, 1:].mean()))
        n = correction['frames']
        result['checkpoints'][tag] = dict(epoch=ckpt['epoch'], sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            implementation_parity_max_error=max_parity,
            matrices={key: stats(value) for key, value in [('pre', a), ('suc', b), ('initial_pre', ai), ('initial_suc', bi)]},
            mean_logit_correction=dict(background=correction['background_sum']/n, foreground=correction['foreground_sum']/n),
            variants={key: dict(**meters[key].scores(), background_fraction=counts[key]['background']/n,
                               changed_fraction=counts[key]['changed']/n) for key in variants})
        print(json.dumps(result['checkpoints'][tag], indent=2), flush=True)
    target = ROOT / args.output
    target.mkdir(parents=True, exist_ok=True)
    (target / 'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

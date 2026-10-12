"""Read saved validation histories only; no model inference or test data."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/online_convergence_review'


def score(row):
    value = row['foreground_canonical']
    return (value['edit'] + value['f1_50']) / 2


def main():
    runs = [('mstcn_pku_dev_seed17', 'mstcn'),
            ('mstcn_pku_dev_e100_resume_seed17', 'mstcn'),
            ('protas_pku_fixedgraph_dev_seed17', 'protas'),
            ('protas_pku_fixedgraph_e100_resume_seed17', 'protas')]
    result = {}
    for name, kind in runs:
        folder = ROOT / 'runs' / name
        path = folder / kind / 'history.json'
        config = json.loads((folder / 'config.json').read_text())
        history = json.loads(path.read_text())
        assert [r['epoch'] for r in history] == list(range(1, history[-1]['epoch'] + 1))
        best = max(history, key=score)
        windows = []
        for lower in range(1, history[-1]['epoch'] + 1, 10):
            rows = [r for r in history if lower <= r['epoch'] < lower + 10]
            window_best = max(rows, key=score)
            windows.append(dict(first=lower, last=rows[-1]['epoch'],
                mean_validation_score=sum(map(score, rows))/len(rows),
                mean_train_loss=sum(r['loss'] for r in rows)/len(rows),
                best_epoch=window_best['epoch'], best_score=score(window_best)))
        result[name] = dict(kind=kind, epochs=history[-1]['epoch'],
            split_sha256=config['split_sha256'], selected=config['selected'],
            history_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            best_epoch=best['epoch'], best_score=score(best), best=best,
            last=history[-1], windows=windows)
    base, continued = [json.loads((ROOT / 'runs' / name / 'mstcn/history.json').read_text())
                      for name in [runs[0][0], runs[1][0]]]
    assert base == continued[:len(base)], 'Continuation does not preserve original history'
    protas_base, protas_continued = [json.loads((ROOT / 'runs' / name / 'protas/history.json').read_text())
                                   for name in [runs[2][0], runs[3][0]]]
    assert protas_base == protas_continued[:len(protas_base)]
    assert len({r['split_sha256'] for r in result.values()}) == 1
    assert all(r['selected'] == result[runs[0][0]]['selected'] for r in result.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'convergence.json').write_text(json.dumps(dict(scope='Saved train/val logs only',
        identical_selected_data=True, mstcn_continuation_prefix_verified=True,
        protas_continuation_prefix_verified=True,
        runs=result), indent=2), encoding='utf-8')
    for name, r in result.items():
        print(name, 'best epoch', r['best_epoch'], 'score', round(r['best_score'], 2))


if __name__ == '__main__':
    main()

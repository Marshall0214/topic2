"""Validation-only descriptive diagnosis; does not tune or modify the model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from baseline import Meter, build_model, load_sequence, predict_logits, segments


def diagnose(pred, y, confusion, boundaries, durations):
    np.add.at(confusion, (y, pred), 1)
    pl, ps, pe = map(np.asarray, segments(pred))
    gl, gs, ge = map(np.asarray, segments(y))
    used = set()
    offsets = []
    missed = fragmented = 0
    for label, start, end in zip(gl, gs, ge):
        durations.append(int(end - start))
        overlap = np.maximum(0, np.minimum(end, pe) - np.maximum(start, ps))
        same = pl == label
        # Count same-class predicted runs inside each GT segment, even across background gaps.
        pieces = int(np.sum((overlap > 0) & same))
        missed += pieces == 0
        fragmented += pieces >= 2
    # Descriptive one-to-one same-class matching, IoU>=0.1; not an official metric.
    candidates = []
    for i, (label, start, end) in enumerate(zip(pl, ps, pe)):
        overlap = np.maximum(0, np.minimum(end, ge) - np.maximum(start, gs))
        union = np.maximum(end, ge) - np.minimum(start, gs)
        ious = overlap / union
        for j in np.flatnonzero((gl == label) & (ious >= .1)):
            candidates.append((float(ious[j]), i, int(j)))
    used_gt = set()
    for iou, i, j in sorted(candidates, reverse=True):
        if i in used or j in used_gt:
            continue
        used.add(i)
        used_gt.add(j)
        offsets.append([int(ps[i]-gs[j]), int(pe[i]-ge[j]), iou])
    transitions = np.flatnonzero(y[1:] != y[:-1]) + 1
    distance = np.full(len(y), len(y))
    for t in transitions:
        distance = np.minimum(distance, np.abs(np.arange(len(y))-t))
    for width in [5, 10, 20]:
        near = distance <= width
        boundaries[width] += [int(near.sum()), int(((pred != y) & near).sum()),
                              int((~near).sum()), int(((pred != y) & ~near).sum())]
    return dict(gt_segments=len(gl), predicted_segments=len(pl),
                gt_without_same_class_overlap=int(missed),
                gt_with_multiple_same_class_runs=int(fragmented), matched_offsets=offsets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='mstcn_pku_dev_seed17')
    args = p.parse_args()
    if Path(args.run).name != args.run:
        p.error('run must be a directory name')
    run = ROOT / 'runs' / args.run
    cfg = json.loads((run / 'config.json').read_text())
    if cfg['normalization'] != 'none' or cfg.get('trainval'):
        raise ValueError('This diagnostic expects the current unnormalized development protocol')
    split_path = Path(cfg['split_path'])
    assert hashlib.sha256(split_path.read_bytes()).hexdigest() == cfg['split_sha256']
    names = cfg['selected']['val']
    assert names and not set(names) & set(cfg['selected']['train'])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = run / 'mstcn/best.pt'
    state = torch.load(checkpoint, map_location='cuda', weights_only=True)
    model = build_model('mstcn', run/'train_graph.pkl', cfg['stages'], cfg['layers'], cfg['width']).cuda().eval()
    model.load_state_dict(state['model'], strict=True)
    confusion = np.zeros((52,52), dtype=np.int64)
    boundaries = {w:np.zeros(4,dtype=np.int64) for w in [5,10,20]}
    meter = Meter()
    rows, offsets, durations = [], [], []
    out = run / 'validation_error_analysis'
    out.mkdir(exist_ok=True)
    cache = out / 'predictions'
    cache.mkdir(exist_ok=True)
    with torch.inference_mode():
        for name in names:
            feat, y = load_sequence(name, Path(cfg['data_root']))
            x = torch.from_numpy(feat)[None].cuda()
            pred = predict_logits(model,'mstcn',x,torch.ones(1,1,len(y),device='cuda')).argmax(1)[0].cpu().numpy()
            np.savez_compressed(cache/(Path(name).stem+'.npz'), pred=pred, target=y)
            meter.update(pred,y)
            a = diagnose(pred,y,confusion,boundaries,durations)
            offsets.extend(a.pop('matched_offsets'))
            rows.append(dict(sequence=name, **a))
    np.savetxt(out/'confusion.csv',confusion,delimiter=',',fmt='%d')
    with (out/'sequences.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    support=confusion.sum(1)
    classes=[]
    for label in np.flatnonzero(support):
        wrong=confusion[label].copy();wrong[label]=0
        classes.append(dict(class_id=int(label),frames=int(support[label]),
                            recall_pct=float(100*confusion[label,label]/support[label]),
                            most_confused_id=int(wrong.argmax()) if wrong.sum() else None,
                            most_confused_frames=int(wrong.max())))
    total=int(confusion.sum());correct=int(np.trace(confusion))
    error=int(total-correct)
    counts={'foreground_to_background':int(confusion[1:,0].sum()),
            'background_to_foreground':int(confusion[0,1:].sum()),
            'foreground_wrong_action':int(confusion[1:,1:].sum()-np.trace(confusion[1:,1:]))}
    pairs=confusion.copy();np.fill_diagonal(pairs,0)
    top=np.argsort(pairs.ravel())[::-1][:15]
    boundary_stats={str(w):dict(near_frames=int(v[0]),near_error_pct=float(100*v[1]/v[0]) if v[0] else None,
                               far_frames=int(v[2]),far_error_pct=float(100*v[3]/v[2]) if v[2] else None,
                               share_of_all_errors_near_pct=float(100*v[1]/error)) for w,v in boundaries.items()}
    off=np.asarray(offsets)
    result=dict(split='validation only',epoch=state['epoch'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                metrics=meter.scores(),frames=total,errors=error,
                error_types={k:dict(frames=v,share_of_errors_pct=100*v/error) for k,v in counts.items()},
                foreground_recall_pct=float(100*np.trace(confusion[1:,1:])/support[1:].sum()),
                predicted_background_pct=float(100*confusion[:,0].sum()/total),
                true_background_pct=float(100*support[0]/total),
                segments={k:sum(r[k] for r in rows) for k in rows[0] if k!='sequence'},
                median_gt_duration_frames=float(np.median(durations)),boundary_neighborhoods=boundary_stats,
                matched_boundary=dict(count=len(offsets),start_signed_median_frames=float(np.median(off[:,0])) if len(off) else None,
                                      end_signed_median_frames=float(np.median(off[:,1])) if len(off) else None,
                                      start_absolute_median_frames=float(np.median(abs(off[:,0]))) if len(off) else None,
                                      end_absolute_median_frames=float(np.median(abs(off[:,1]))) if len(off) else None),
                classes=classes,top_confusions=[dict(target=int(i//52),predicted=int(i%52),frames=int(pairs.ravel()[i])) for i in top if pairs.ravel()[i]],
                limitations=['Same-class IoU>=0.1 boundary matching is descriptive and conditional on successful matches; misses reported separately.',
                             'Boundary offsets are frames, not milliseconds; signed positive means later.',
                             'Multiple predicted runs can represent short interruptions or wrong-class runs; not necessarily many extra full actions.',
                             'No test predictions used for selecting a method or hyperparameters.'])
    (out/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['classes']},indent=2))


if __name__=='__main__':
    main()

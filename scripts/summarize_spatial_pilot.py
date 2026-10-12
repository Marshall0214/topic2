"""Summarize frozen pilot and diagnose validation-only best checkpoints."""
import hashlib
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import load_sequence,predict_logits,segments,Meter
from spatial_encoders import build_spatial_model

torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
protocol=json.loads((ROOT/'configs/spatial_pilot.json').read_text())
parser=argparse.ArgumentParser()
parser.add_argument('--epochs',type=int,choices=[30,60],default=30)
args=parser.parse_args()
protocol['run_prefix']=protocol['run_prefix'].replace('_e30_',f'_e{args.epochs}_')
protocol['epochs']=args.epochs
rows=[]; selected=None
for encoder in protocol['encoders']:
    run=ROOT/'runs'/(protocol['run_prefix']+'_'+encoder)
    cfg=json.loads((run/'config.json').read_text())
    history=json.loads((run/'mstcn/history.json').read_text())
    assert len(history)==protocol['epochs'],'Pilot training not complete'
    if selected is None:selected=cfg['selected']
    assert cfg['selected']==selected
    assert not cfg['trainval'] and not cfg['test_used']
    assert not set(selected['train'])&set(selected['val'])
    hashes={k.replace('\\','/'):v for k,v in cfg['source_hashes'].items()}
    assert hashes['src/baseline.py']==hashlib.sha256((ROOT/'src/baseline.py').read_bytes()).hexdigest()
    assert hashes['src/spatial_encoders.py']==hashlib.sha256((ROOT/'src/spatial_encoders.py').read_bytes()).hexdigest()
    state=torch.load(run/'mstcn/best.pt',map_location='cuda',weights_only=True)
    best=next(r for r in history if r['epoch']==state['epoch'])
    model=build_spatial_model(encoder,cfg['stages'],cfg['layers'],cfg['width']).cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    confusion=np.zeros((52,52),dtype=np.int64)
    total_segments=missed=fragmented=0
    meter=Meter()
    with torch.inference_mode():
        for name in selected['val']:
            feat,y=load_sequence(name,Path(cfg['data_root']))
            x=torch.from_numpy(feat)[None].cuda()
            pred=predict_logits(model,'mstcn',x,torch.ones(1,1,len(y),device='cuda')).argmax(1)[0].cpu().numpy()
            meter.update(pred,y)
            np.add.at(confusion,(y,pred),1)
            pl,ps,pe=map(np.asarray,segments(pred))
            for c,s,e in zip(*segments(y)):
                overlaps=(pl==c)&(np.minimum(e,pe)>np.maximum(s,ps))
                n=int(overlaps.sum());total_segments+=1;missed+=n==0;fragmented+=n>=2
    assert all(abs(meter.scores()[k]-best['foreground_canonical'][k])<1e-8 for k in ['accuracy','edit','f1_10','f1_25','f1_50','avg'])
    rows.append(dict(encoder=encoder,best_epoch=state['epoch'],parameters=sum(p.numel() for p in model.parameters()),
                     best_validation=best['foreground_canonical'],last_validation=history[-1]['foreground_canonical'],
                     foreground_accuracy_pct=float(100*np.trace(confusion[1:,1:])/confusion[1:,:].sum()),
                     missed_segments=missed,total_segments=total_segments,fragmented_segments=fragmented,
                     mean_epoch_seconds=float(np.mean([r['seconds'] for r in history])),
                     predicted_background_pct=float(100*confusion[:,0].sum()/confusion.sum()),
                     checkpoint_sha256=hashlib.sha256((run/'mstcn/best.pt').read_bytes()).hexdigest()))
out=ROOT/('reports/spatial_pilot' if args.epochs==30 else 'reports/spatial_pilot_e60')
out.mkdir(exist_ok=True)
result=dict(protocol=protocol,train_sequences=len(selected['train']),val_sequences=len(selected['val']),
            test_evaluated=False,selection_verified_identical=True,results=rows,
            limitations=['Single seed and subset; no significance or full-data superiority claims.',
                         'MLP/GCN approximately parameter-matched; compute differs.',
                         'Missing segment means no same-class overlap, not official IoU@50 false negative.',
                         'Epoch timings include training AND validation; not inference latency.'])
(out/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

# Static export for comparing convergence; all curves use the same validation subset.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(12,3.4))
for encoder in protocol['encoders']:
    h=json.loads((ROOT/'runs'/(protocol['run_prefix']+'_'+encoder)/'mstcn/history.json').read_text())
    epochs=[r['epoch'] for r in h]
    axes[0].plot(epochs,[r['loss'] for r in h],label=encoder)
    axes[1].plot(epochs,[r['foreground_canonical']['edit'] for r in h],label=encoder)
    axes[2].plot(epochs,[r['foreground_canonical']['f1_50'] for r in h],label=encoder)
for ax,title in zip(axes,['Training loss','Validation Edit','Validation F1@50']):
    ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2);ax.legend()
fig.suptitle('Spatial pilot: same 128 training / 24 validation sequences, seed 17')
fig.tight_layout()
fig.savefig(out/'learning_curves.png',dpi=180)
fig.savefig(out/'learning_curves.svg')
plt.close(fig)

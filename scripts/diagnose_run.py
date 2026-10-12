"""Inspect saved checkpoints, class collapse and gradients without test data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import load_sequence,build_model,predict_logits
from normalization import TrainingStandardizer,feature_validity

parser=argparse.ArgumentParser()
parser.add_argument('--run',default='small24_val8_e10_seed17')
args=parser.parse_args()
run=ROOT/'runs'/args.run
config=json.loads((run/'config.json').read_text())
torch.set_num_threads(4)
device='cuda' if torch.cuda.is_available() else 'cpu'
cache={n:load_sequence(n) for part in ['train','val'] for n in config['selected'][part]}
if config.get('normalization')=='train-standard':
    standardizer=TrainingStandardizer.load(run/'normalization.npz')
    root=Path(config['data_root'])
    cache={n:(standardizer.transform(x,feature_validity(np.load(root/'features'/n,allow_pickle=False))),y) for n,(x,y) in cache.items()}
report={'run':args.run,'test_used':False,'data':{},'models':{}}
for part in ['train','val']:
    ys=np.concatenate([cache[n][1] for n in config['selected'][part]])
    report['data'][part]={'frames':len(ys),'background_fraction':float((ys==0).mean()),'class_counts':dict(sorted(Counter(map(int,ys)).items()))}
for kind in ['mstcn','protas']:
    if not (run/kind/'last.pt').exists():
        continue
    model=build_model(kind,run/'train_graph.pkl',config['stages'],config['layers'],config['width']).to(device).eval()
    details={}
    for checkpoint_name in ['best','last']:
        ckpt=torch.load(run/kind/f'{checkpoint_name}.pt',map_location=device,weights_only=True)
        model.load_state_dict(ckpt['model'])
        checks={}
        with torch.inference_mode():
            for part in ['train','val']:
                predicted,stage_counts={},[Counter() for _ in range(config['stages'])]
                counts=Counter()
                correct=total=fgcorrect=fgframes=0
                bg_prob=0.
                for name in config['selected'][part]:
                    features,y=cache[name]
                    x=torch.from_numpy(features)[None].to(device)
                    mask=torch.ones(1,1,len(y),device=device)
                    outputs,progress=model(x,mask)
                    logits=outputs[-1] if kind=='mstcn' else model.graph_learner.inference(outputs[-1],progress[-1])
                    pred=logits.argmax(1)[0].cpu().numpy()
                    counts.update(map(int,pred))
                    for counter,stage in zip(stage_counts,outputs):
                        counter.update(map(int,stage.argmax(1)[0].cpu().numpy()))
                    correct+=int((pred==y).sum())
                    total+=len(y)
                    fgframes+=int((y!=0).sum())
                    fgcorrect+=int(((pred==y)&(y!=0)).sum())
                    bg_prob+=float(logits.softmax(1)[0,0].sum())
                checks[part]=dict(accuracy=100*correct/total,foreground_accuracy=100*fgcorrect/max(1,fgframes),predicted_background_fraction=counts[0]/total,mean_background_probability=bg_prob/total,prediction_counts=dict(counts),stage_prediction_counts=[dict(c) for c in stage_counts])
        details[checkpoint_name]=dict(epoch=ckpt['epoch'],checks=checks)
    first=torch.load(run/kind/'best.pt',map_location='cpu',weights_only=True)['model']
    last=torch.load(run/kind/'last.pt',map_location='cpu',weights_only=True)['model']
    details['weight_change_l2']=sum(float((last[k]-v).float().square().sum()) for k,v in first.items())**.5
    report['models'][kind]=details
output=ROOT/'reports'/f'{args.run}_diagnosis.json'
output.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({'data':{p:{k:v for k,v in d.items() if k!='class_counts'} for p,d in report['data'].items()},'models':{m:{'weight_change_l2':r['weight_change_l2'],'last':{p:{k:v for k,v in d.items() if k not in ['stage_prediction_counts','prediction_counts']} for p,d in r['last']['checks'].items()}} for m,r in report['models'].items()}},indent=2))

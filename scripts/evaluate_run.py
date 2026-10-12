"""Evaluate an explicitly selected checkpoint; never choose by test score."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import Meter,OfficialMeter,build_model,load_sequence,predict_logits
from normalization import TrainingStandardizer,feature_validity
from spatial_encoders import build_spatial_model

parser=argparse.ArgumentParser()
parser.add_argument('--run',required=True)
parser.add_argument('--model',choices=['mstcn','protas','both'],default='both')
parser.add_argument('--checkpoint',choices=['best','last'],default='last')
parser.add_argument('--split',choices=['val','test'],default='val')
parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
parser.add_argument('--output-suffix',default='',help='Save supplemental evaluation without replacing earlier files')
args=parser.parse_args()
if Path(args.run).name!=args.run:
    parser.error('run must be a directory name')
if args.output_suffix and not all(c.isalnum() or c=='_' for c in args.output_suffix):
    parser.error('Output suffix must be alphanumeric/underscore')
run=ROOT/'runs'/args.run
config=json.loads((run/'config.json').read_text())
if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
split_path=Path(config['split_path'])
if hashlib.sha256(split_path.read_bytes()).hexdigest()!=config['split_sha256']:
    raise RuntimeError('Split file changed since training')
split=json.loads(split_path.read_text())['splits']
names=split['test'] if args.split=='test' else config['selected']['val']
if not names:raise RuntimeError('This final trainval run has no validation set; evaluate fixed last checkpoint on test explicitly.')
if config.get('trainval') and args.checkpoint!='last':raise RuntimeError('Final trainval protocol only has fixed last checkpoint.')
if set(names)&set(config['selected']['train']):raise RuntimeError('Train/evaluation overlap')
standardizer=TrainingStandardizer.load(run/'normalization.npz') if config.get('normalization')=='train-standard' else None
output={'run':args.run,'evaluation_split':args.split,'checkpoint_rule':args.checkpoint,'sequences':names,
        'paper_comparability':'Local reference protocol; Table 7 exact input,metric and checkpoint details remain unverified.',
        'models':{}}
for kind in (['mstcn','protas'] if args.model=='both' else [args.model]):
    path=run/kind/f'{args.checkpoint}.pt'
    ckpt=torch.load(path,map_location=args.device,weights_only=True)
    mode=config.get('graph_reference','independent_copy')
    model=(build_spatial_model(config.get('spatial_encoder','linear'),config['stages'],config['layers'],config['width']) if kind=='mstcn' else build_model(kind,run/'train_graph.pkl',config['stages'],config['layers'],config['width'],graph_reference=mode,task_graph=config.get('task_graph','learnable'))).to(args.device).eval()
    model.load_state_dict(ckpt['model'],strict=True)
    meters={'foreground_canonical':Meter(),'dest_legacy':Meter(legacy=True),'protas_official':OfficialMeter()}
    rows=[]
    with torch.inference_mode():
        for i,name in enumerate(names):
            feat,y=load_sequence(name,Path(config['data_root']))
            if standardizer:
                raw=np.load(Path(config['data_root'])/'features'/name,allow_pickle=False)
                feat=standardizer.transform(feat,feature_validity(raw))
            x=torch.from_numpy(feat)[None].to(args.device)
            pred=predict_logits(model,kind,x,torch.ones(1,1,len(y),device=args.device)).argmax(1)[0].cpu().numpy()
            for meter in meters.values():meter.update(pred,y)
            one=Meter();one.update(pred,y)
            rows.append(dict(name=name,**one.scores()))
            if (i+1)%25==0:print(f'{kind}/{args.split}: {i+1}/{len(names)}',flush=True)
    output['models'][kind]={'epoch':ckpt['epoch'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'scores':{name:meter.scores() for name,meter in meters.items()},'per_sequence':rows}
    print(json.dumps({kind:output['models'][kind]['scores']},indent=2),flush=True)
suffix=f'_{args.output_suffix}' if args.output_suffix else ''
target=run/f'{args.split}_{args.checkpoint}_evaluation{suffix}.json'
target.write_text(json.dumps(output,indent=2),encoding='utf-8')
print(f'Saved: {target}',flush=True)

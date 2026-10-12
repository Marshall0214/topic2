"""Read-only failure breakdown, raw annotation alignment and feature audit."""
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import DATA,Meter,build_model,load_sequence,predict_logits,segments
RUN=ROOT/'runs/small24_val8_e50_seed17'
RAW=Path(r'D:\AQA\dataset\PKUMMDv2')
OUT=ROOT/'reports/failure_analysis50'
OUT.mkdir(parents=True,exist_ok=True)
config=json.loads((RUN/'config.json').read_text())
torch.set_num_threads(4)
device='cuda' if torch.cuda.is_available() else 'cpu'
cache={n:load_sequence(n) for part in ['train','val'] for n in config['selected'][part]}
result={'run':str(RUN),'test_used':False,'features':{},'alignment':{},'models':{}}
feature_rows=[]
alignments=[]
for part in ['train','val']:
    norms,motions,features=[],[],[]
    for name in config['selected'][part]:
        feat,y=cache[name]
        raw=np.load(DATA/'features'/name,allow_pickle=False)
        p=raw.reshape(-1,2,25,3)
        valid=np.any(p!=0,axis=(2,3))
        norm=np.linalg.norm(feat[:150].T.reshape(-1,2,25,3),axis=-1)
        motion=np.linalg.norm(feat[150:].T.reshape(-1,2,25,3),axis=-1)
        feature_rows.append(dict(split=part,name=name,view=name[-5],frames=len(y),body1_valid=float(valid[:,0].mean()),body2_valid=float(valid[:,1].mean()),relative_norm_p99=float(np.quantile(norm,.99)),motion_norm_p99=float(np.quantile(motion,.99)),motion_max=float(motion.max())))
        norms.append(norm.flatten());motions.append(motion.flatten());features.append(feat)
        raw_path=RAW/'skeletons'/Path(name).with_suffix('.txt').name
        annotation_path=RAW/'label'/raw_path.name
        raw_input=np.loadtxt(raw_path)
        ann=np.loadtxt(annotation_path,delimiter=',',ndmin=2,dtype=int)
        row={'split':part,'name':name,'raw_shape':list(raw_input.shape),'raw_matches':raw_input.shape==raw.shape and bool(np.allclose(raw_input,raw,atol=1e-6,rtol=1e-6)),'label_agreements':{}}
        for convention,shift,end_shift in [('zero_half_open',0,0),('one_inclusive',-1,0),('zero_inclusive',0,1)]:
            for fill in [0,5,10,15,20]:
                expected=np.zeros(len(y),dtype=np.int64)
                ordered=ann[np.argsort(ann[:,1],kind='stable')]
                for i,a in enumerate(ordered):
                    label,start,end=a[:3]
                    start,end=start+shift,end+end_shift
                    if i+1<len(ordered):
                        following=int(ordered[i+1,1])+shift
                        if 0<=following-end<=fill:
                            end=following
                    expected[max(0,start):min(len(y),end)]=label
                row['label_agreements'][f'{convention}_fill{fill}']=dict(matched=int((expected==y).sum()),frames=len(y))
        alignments.append(row)
    f=np.concatenate(features,axis=1)
    result['features'][part]={'frames':f.shape[1],'mean':f.mean(1).tolist(),'std':f.std(1).tolist(),
        'relative_norm_quantiles':np.quantile(np.concatenate(norms),[.5,.9,.99,1]).tolist(),
        'motion_norm_quantiles':np.quantile(np.concatenate(motions),[.5,.9,.99,1]).tolist()}
train_mean=np.asarray(result['features']['train']['mean'])
train_std=np.asarray(result['features']['train']['std'])
val_mean=np.asarray(result['features']['val']['mean'])
result['features']['mean_shift_in_train_std']={'median':float(np.median(np.abs(val_mean-train_mean)/np.maximum(train_std,.01))),'p95':float(np.quantile(np.abs(val_mean-train_mean)/np.maximum(train_std,.01),.95))}
result['alignment']={'sequences':len(alignments),'raw_matches_all':all(a['raw_matches'] for a in alignments),'aggregate':{}}
for key in alignments[0]['label_agreements']:
    matches=sum(a['label_agreements'][key]['matched'] for a in alignments)
    total=sum(a['label_agreements'][key]['frames'] for a in alignments)
    result['alignment']['aggregate'][key]={'agreement':matches/total,'mismatched_frames':total-matches}
(OUT/'alignment_details.json').write_text(json.dumps(alignments,indent=2),encoding='utf-8')
with (OUT/'feature_sequences.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=list(feature_rows[0]));writer.writeheader();writer.writerows(feature_rows)

mapping={int(line.split(maxsplit=1)[0]):line.split(maxsplit=1)[1] for line in (DATA/'mapping.txt').read_text().splitlines()}
for kind in ['mstcn','protas']:
    ckpt=torch.load(RUN/kind/'best.pt',map_location=device,weights_only=True)
    model=build_model(kind,RUN/'train_graph.pkl').to(device).eval()
    model.load_state_dict(ckpt['model'])
    details={'epoch':ckpt['epoch'],'splits':{}}
    sequence_rows=[]
    for part in ['train','val']:
        meter=Meter();views=defaultdict(Meter);confusion=np.zeros((52,52),dtype=np.int64)
        predicted_segments=true_segments=0
        with torch.inference_mode():
            for name in config['selected'][part]:
                feat,y=cache[name]
                x=torch.from_numpy(feat)[None].to(device)
                pred=predict_logits(model,kind,x,torch.ones(1,1,len(y),device=device)).argmax(1)[0].cpu().numpy()
                meter.update(pred,y);views[name[-5]].update(pred,y)
                confusion+=np.bincount(y*52+pred,minlength=52*52).reshape(52,52)
                pseg=len(segments(pred)[0]);gseg=len(segments(y)[0])
                predicted_segments+=pseg;true_segments+=gseg
                one=Meter();one.update(pred,y)
                sequence_rows.append(dict(split=part,name=name,view=name[-5],predicted_segments=pseg,true_segments=gseg,
                    predicted_background=float((pred==0).mean()),foreground_accuracy=100*float(((pred==y)&(y!=0)).sum())/max(1,int((y!=0).sum())),**one.scores()))
        rows=[]
        for label in np.flatnonzero(confusion.sum(1)):
            counts=confusion[label];total=int(counts.sum());best=int(counts.argmax())
            rows.append(dict(id=int(label),name=mapping[int(label)],frames=total,recall=100*float(counts[label])/total,top_prediction=best,top_prediction_name=mapping[best],background_fraction=float(counts[0])/total))
        details['splits'][part]={'scores':meter.scores(),'views':{v:m.scores() for v,m in views.items()},'predicted_segments':predicted_segments,'true_segments':true_segments,'classes':rows}
        np.save(OUT/f'{kind}_{part}_confusion.npy',confusion)
    with (OUT/f'{kind}_sequences.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(sequence_rows[0]));writer.writeheader();writer.writerows(sequence_rows)
    result['models'][kind]=details
    print(f'{kind}: train={details["splits"]["train"]["scores"]}; val={details["splits"]["val"]["scores"]}',flush=True)
(OUT/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'alignment':result['alignment'],'feature_quantiles':{p:{k:v for k,v in d.items() if k not in ['mean','std']} for p,d in result['features'].items()}},indent=2))

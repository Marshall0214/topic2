"""Paired validation diagnosis from saved predictions; no test or tuning."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import segments

runs=['mstcn_pku_dev_seed17','protas_pku_fixedgraph_dev_seed17']
folder='validation_pair_fixedgraph_v1'
dirs=[ROOT/'runs'/run/folder for run in runs]
cfgs=[json.loads((ROOT/'runs'/run/'config.json').read_text()) for run in runs]
assert cfgs[0]['selected']['val']==cfgs[1]['selected']['val']
assert cfgs[0]['split_sha256']==cfgs[1]['split_sha256']
data=[json.loads((path/'diagnosis.json').read_text()) for path in dirs]
names=cfgs[0]['selected']['val']
bins={key:dict(segments=0,missed=[0,0],fragmented=[0,0]) for key in ['1-30','31-60','61-120','121+']}
paired=dict(both_missed=0,only_mstcn_missed=0,only_protas_missed=0,neither_missed=0)
rows=[];common_offsets=[[],[]]
def matched_offsets(pred,y):
    pl,ps,pe=map(np.asarray,segments(pred));gl,gs,ge=map(np.asarray,segments(y))
    candidates=[]
    for i,(label,start,end) in enumerate(zip(pl,ps,pe)):
        iou=np.maximum(0,np.minimum(end,ge)-np.maximum(start,gs))/(np.maximum(end,ge)-np.minimum(start,gs))
        candidates.extend((float(iou[j]),i,int(j)) for j in np.flatnonzero((gl==label)&(iou>=.1)))
    used=set();result={}
    for iou,i,j in sorted(candidates,reverse=True):
        if i in used or j in result:continue
        used.add(i);result[j]=[int(ps[i]-gs[j]),int(pe[i]-ge[j])]
    return result
for name in names:
    arrays=[np.load(path/'predictions'/(Path(name).stem+'.npz')) for path in dirs]
    y=arrays[0]['target'];assert np.array_equal(y,arrays[1]['target'])
    preds=[a['pred'] for a in arrays]
    gl,gs,ge=segments(y)
    segs=[tuple(map(np.asarray,segments(pred))) for pred in preds]
    matches=[matched_offsets(pred,y) for pred in preds]
    for j in set(matches[0])&set(matches[1]):
        for k in range(2):common_offsets[k].append(matches[k][j])
    for j,(label,start,end) in enumerate(zip(gl,gs,ge)):
        duration=end-start
        key='1-30' if duration<=30 else '31-60' if duration<=60 else '61-120' if duration<=120 else '121+'
        bins[key]['segments']+=1
        pieces=[]
        for k,(pl,ps,pe) in enumerate(segs):
            count=int(((pl==label)&(np.minimum(end,pe)-np.maximum(start,ps)>0)).sum())
            pieces.append(count);bins[key]['missed'][k]+=int(count==0);bins[key]['fragmented'][k]+=int(count>=2)
        category=('both_missed' if pieces==[0,0] else 'only_mstcn_missed' if pieces[0]==0 else 'only_protas_missed' if pieces[1]==0 else 'neither_missed')
        paired[category]+=1
        rows.append(dict(sequence=name,class_id=label,start=start,end=end,duration=duration,category=category,pieces=pieces))
    for a in arrays:a.close()
summary=dict(scope='77 validation sequences; best checkpoints; descriptive diagnosis only',runs=runs,
             diagnoses=data,duration_bins=bins,paired_misses=paired,common_matched_segments=len(common_offsets[0]),
             common_boundary_offsets={run:dict(start_signed_median=float(np.median(np.asarray(offsets)[:,0])),
                 end_signed_median=float(np.median(np.asarray(offsets)[:,1])),
                 start_absolute_median=float(np.median(np.abs(np.asarray(offsets)[:,0]))),
                 end_absolute_median=float(np.median(np.abs(np.asarray(offsets)[:,1]))))
                 for run,offsets in zip(runs,common_offsets)})
out=ROOT/'reports/protas_fixed_graph/validation_comparison'
out.mkdir(parents=True,exist_ok=True)
(out/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
(out/'segments.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
# Deterministic examples: earliest short ProTAS-only misses, then MS-TCN-only misses.
chosen=[]
for category in ['only_protas_missed','only_mstcn_missed']:
    chosen.extend(sorted((r for r in rows if r['category']==category),key=lambda r:(r['duration'],r['sequence'],r['start']))[:3])
html=['<!doctype html><meta charset="utf-8"><title>验证集动作漏检对照</title><style>body{font:16px sans-serif;max-width:1100px;margin:30px auto}svg{width:100%;height:125px}p{line-height:1.6}</style><h1>验证集动作漏检对照</h1><p>上到下：真值、MS-TCN、固定图 ProTAS。灰色为背景；相同类别同色。标注框为待检查真值动作。按时长排序选例，不代表总体表现。</p>']
for r in chosen:
    name=r['sequence'];start=max(0,r['start']-60)
    arrays=[np.load(path/'predictions'/(Path(name).stem+'.npz')) for path in dirs]
    end=min(len(arrays[0]['target']),r['end']+60)
    html.append(f'<h3>{name}：类别{r["class_id"]}，{r["category"]}，时长{r["duration"]}帧</h3><svg viewBox="0 0 1000 125">')
    for i,seq in enumerate([arrays[0]['target'],arrays[0]['pred'],arrays[1]['pred']]):
        labels,ss,ee=segments(seq[start:end],exclude_background=False)
        for label,s,e in zip(labels,ss,ee):
            color='#ccc' if label==0 else f'hsl({label*137%360} 65% 55%)'
            x=150+850*s/(end-start);width=850*(e-s)/(end-start)
            html.append(f'<rect x="{x}" y="{i*38}" width="{width}" height="28" fill="{color}"><title>类别{label}，帧{start+s}—{start+e}</title></rect>')
        html.append(f'<text x="0" y="{i*38+20}">{["真值","MS-TCN","固定图 ProTAS"][i]}</text>')
    html.append(f'<rect x="{150+850*(r["start"]-start)/(end-start)}" y="0" width="{850*r["duration"]/(end-start)}" height="108" fill="none" stroke="black" stroke-width="2"/></svg>')
    for a in arrays:a.close()
(out/'cases.html').write_text('\n'.join(html),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='diagnoses'},indent=2))

"""Inspect cached validation errors and raw skeletons, without model changes."""
import csv
import html
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import segments

EDGES=[(0,1),(1,20),(20,2),(2,3),(20,4),(4,5),(5,6),(6,7),(7,21),(6,22),
       (20,8),(8,9),(9,10),(10,11),(11,23),(10,24),(0,12),(12,13),(13,14),(14,15),
       (0,16),(16,17),(17,18),(18,19)]


def timeline(y,pred):
    width=1100
    svg=['<svg viewBox="0 0 1200 110" role="img">']
    for row,(label,values) in enumerate([('GT',y),('Prediction',pred)]):
        svg.append(f'<text x="0" y="{25+row*40}">{label}</text>')
        labels,starts,ends=segments(values,exclude_background=False)
        for c,s,e in zip(labels,starts,ends):
            color='#ddd' if c==0 else f'hsl({int(c)*137%360},60%,55%)'
            x=90+width*s/len(y); w=width*(e-s)/len(y)
            svg.append(f'<rect x="{x}" y="{row*40+5}" width="{w}" height="28" fill="{color}"><title>class {c}: [{s},{e})</title></rect>')
            if w>18:svg.append(f'<text x="{x+2}" y="{row*40+25}" font-size="12">{c}</text>')
    svg.append(f'<text x="90" y="100">0</text><text x="1100" y="100">{len(y)} frames</text></svg>')
    return ''.join(svg)


def skeleton(raw,frame):
    p=raw[frame].reshape(2,25,3)
    svg=['<svg viewBox="0 0 640 230">']
    for k,(a,b) in enumerate([(0,1),(0,2),(2,1)]):
        valid=[v for v in p if np.any(v)]
        if not valid:continue
        pts=np.concatenate(valid)[:,[a,b]]
        low=pts.min(0); span=max(float(np.ptp(pts,axis=0).max()),1e-6)
        for body,points in enumerate(p):
            if not np.any(points):continue
            q=(points[:,[a,b]]-low)/span*165
            q[:,0]+=k*210+15;q[:,1]=205-q[:,1]
            for i,j in EDGES:
                svg.append(f'<line x1="{q[i,0]}" y1="{q[i,1]}" x2="{q[j,0]}" y2="{q[j,1]}" stroke="{["#167ac6","#e06518"][body]}" stroke-width="2"/>')
        svg.append(f'<text x="{k*210+15}" y="20">coordinates {a}/{b}, frame {frame}</text>')
    svg.append('</svg>')
    return ''.join(svg)


def main():
    run=ROOT/'runs/mstcn_pku_dev_seed17'
    cfg=json.loads((run/'config.json').read_text())
    out=run/'validation_error_analysis'
    data=Path(cfg['data_root'])
    train_frames=np.zeros(52,dtype=np.int64); train_segments=np.zeros(52,dtype=np.int64)
    for name in cfg['selected']['train']:
        y=np.load(data/'gt_arr'/name,allow_pickle=False)
        train_frames+=np.bincount(y.astype(int),minlength=52)
        for c in segments(y)[0]:train_segments[int(c)]+=1
    records=[]; all_segments=[]; views={}; valid_stats=[]
    for name in cfg['selected']['val']:
        cache=np.load(out/'predictions'/(Path(name).stem+'.npz'))
        y,pred=cache['target'],cache['pred']
        raw=np.load(data/'features'/name,allow_pickle=False)
        assert len(y)==len(raw) and np.isfinite(raw).all()
        points=raw.reshape(-1,2,25,3)
        valid=np.any(points!=0,axis=(2,3))
        both_zero=~valid.any(1)
        relative=points[:,0]-points[:,0,:1]
        delta=np.zeros(len(y));root_delta=np.zeros(len(y))
        adjacent=valid[1:,0]&valid[:-1,0]
        delta[1:]=np.median(np.linalg.norm(np.diff(relative,axis=0),axis=2),axis=1)*adjacent
        root_delta[1:]=np.linalg.norm(np.diff(points[:,0,0],axis=0),axis=1)*adjacent
        valid_stats.append(dict(sequence=name,frames=len(y),body1_missing_frames=int((~valid[:,0]).sum()),
                                body2_present_frames=int(valid[:,1].sum()),both_missing_frames=int(both_zero.sum()),
                                missing_foreground_frames=int((both_zero&(y!=0)).sum()),
                                missing_background_frames=int((both_zero&(y==0)).sum()),
                                errors_on_missing_frames=int((both_zero&(pred!=y)).sum()),
                                zero_joints_in_present_body1=int(((points[:,0]==0).all(2)&valid[:,0,None]).sum()),
                                relative_displacement_p99=float(np.quantile(delta,.99)),root_displacement_p99=float(np.quantile(root_delta,.99))))
        pl,ps,pe=map(np.asarray,segments(pred)); gl,gs,ge=map(np.asarray,segments(y))
        missed=fragmented=0
        for c,s,e in zip(gl,gs,ge):
            overlaps=(pl==c)&(np.minimum(e,pe)>np.maximum(s,ps))
            n=int(overlaps.sum());missed+=n==0;fragmented+=n>=2
            item=dict(sequence=name,class_id=int(c),start=int(s),end=int(e),duration=int(e-s),
                      missed=n==0,fragmented=n>=2,body1_missing_pct=float(100*(~valid[s:e,0]).mean()),
                      both_missing_pct=float(100*both_zero[s:e].mean()),relative_displacement_mean=float(delta[s:e].mean()),
                      root_displacement_mean=float(root_delta[s:e].mean()),train_class_segments=int(train_segments[c]))
            all_segments.append(item)
        fg=y!=0
        wrong=((pred!=y)&fg&(pred!=0)).sum()
        records.append(dict(sequence=name,missed=int(missed),fragmented=int(fragmented),gt_segments=len(gl),
                            foreground_confusion_rate=float(wrong/max(int(fg.sum()),1)),accuracy=float((pred==y).mean())))
        views[name]=(raw,y,pred)
    chosen=[];used=set()
    # Two distinct sequences per failure mode, descending score then filename.
    for category,key in [('missed','missed'),('confusion','foreground_confusion_rate'),('fragmentation','fragmented')]:
        for row in sorted(records,key=lambda r:(-r[key],r['sequence'])):
            if row['sequence'] not in used:
                chosen.append(dict(category=category,**row));used.add(row['sequence'])
                if sum(a['category']==category for a in chosen)==2:break
    group={}
    for label,mask in [('missed',lambda s:s['missed']),('with_same_class_overlap',lambda s:not s['missed'])]:
        items=[s for s in all_segments if mask(s)]
        group[label]={'count':len(items)}
        for key in ['duration','body1_missing_pct','both_missing_pct','relative_displacement_mean','root_displacement_mean','train_class_segments']:
            group[label][key+'_median']=float(np.median([s[key] for s in items]))
    summary=dict(validation_sequences=len(records),input_quality=valid_stats,
                 totals={k:sum(r[k] for r in valid_stats) for k in ['frames','body1_missing_frames','body2_present_frames','both_missing_frames','zero_joints_in_present_body1','missing_foreground_frames','missing_background_frames','errors_on_missing_frames']},
                 missed_segments_without_missing_frames=sum(s['missed'] and s['both_missing_pct']==0 for s in all_segments),
                 segment_groups=group,representatives=chosen,
                 caveats=['Representatives are ranked failure cases, not random or prevalence estimates.',
                          'Displacement is in dataset coordinate units; no physical noise threshold or causal attribution.',
                          'Input check detects missing/zero/nonfinite data but cannot certify pose or annotation accuracy.',
                          'Group comparisons are descriptive and confounded by action class and motion complexity.'])
    (out/'input_inspection.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'segment_input_quality.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(all_segments[0]));w.writeheader();w.writerows(all_segments)
    doc=['<!doctype html><meta charset="utf-8"><title>Validation failure inspection</title><style>body{font:16px sans-serif;max-width:1250px;margin:30px auto}svg{width:100%}section{border-top:1px solid #aaa;margin-top:25px}pre{white-space:pre-wrap}</style>',
         '<h1>验证集失败案例与骨架检查</h1><p>每类按错误数量/比例排序选两条不同序列；非随机样本。灰色为背景，数字为类别ID。骨架为原始坐标三种投影，蓝/橙为两个身体槽位。所有时间为帧索引。</p>']
    for row in chosen:
        name=row['sequence'];raw,y,pred=views[name]
        doc.append(f'<section><h2>{html.escape(name)} — {row["category"]}</h2><pre>{html.escape(json.dumps(row,ensure_ascii=False))}</pre>')
        doc.append(timeline(y,pred))
        candidates=[s for s in all_segments if s['sequence']==name]
        if row['category']=='missed':candidates.sort(key=lambda s:(not s['missed'],s['start']))
        elif row['category']=='fragmentation':candidates.sort(key=lambda s:(not s['fragmented'],s['start']))
        else:candidates.sort(key=lambda s:float((pred[s['start']:s['end']] == s['class_id']).mean()))
        seg=candidates[0];mid=(seg['start']+seg['end'])//2
        doc.append(f'<p>选取动作段 [{seg["start"]},{seg["end"]})，真值 {seg["class_id"]}；中点预测 {pred[mid]}。</p>')
        for frame in [seg['start'],mid,seg['end']-1]:doc.append(skeleton(raw,frame))
        doc.append('</section>')
    (out/'cases.html').write_text(''.join(doc),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='input_quality'},indent=2))


if __name__=='__main__':main()

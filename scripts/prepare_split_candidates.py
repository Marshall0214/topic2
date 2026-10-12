"""Write lists only: upstream compatibility and recording-group holdout."""
import json
import random
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
audit=json.loads((ROOT/'reports/data_audit.json').read_text())
original={p:audit['split_files'][p]['names'] for p in ['train','val','test']}
pool=sorted(set(original['train']+original['val']))
groups=sorted({n.split('-')[0] for n in pool})
random.Random(17).shuffle(groups)
data=Path(audit['data_root'])
group_labels={g:set() for g in groups}
for n in pool:
    group_labels[n.split('-')[0]].update(map(int,np.unique(np.load(data/'gt_arr'/n,allow_pickle=False))))
# Use only development-pool labels to cover all observed classes in validation.
# Stable shuffled order breaks ties; test labels never influence membership.
uncovered=set().union(*group_labels.values())
val_groups=set()
while uncovered:
    chosen=max((g for g in groups if g not in val_groups),key=lambda g:len(group_labels[g]&uncovered))
    val_groups.add(chosen)
    uncovered-=group_labels[chosen]
for g in groups:
    if len(val_groups)>=26:
        break
    val_groups.add(g)
grouped={'train':[n for n in pool if n.split('-')[0] not in val_groups], 'val':[n for n in pool if n.split('-')[0] in val_groups], 'test':original['test']}
assert not({n.split('-')[0] for n in grouped['train']} & val_groups)
OUT=ROOT/'configs'
OUT.mkdir(exist_ok=True)
summary={}
for name,splits in [('dest_compat',original),('recording_holdout_seed17',grouped)]:
    (OUT/f'{name}.json').write_text(json.dumps({'name':name,'splits':splits,'notes':'Only lists; original data and upstream CSV unchanged. Group holdout uses development-pool class coverage and seeded ties; no test data selects membership.'},indent=2),encoding='utf-8')
    detail={}
    for part,names in splits.items():
        labels=set()
        for n in names:
            labels.update(map(int,np.unique(np.load(data/'gt_arr'/n,allow_pickle=False))))
        detail[part]={'sequences':len(names),'recordings':len({n.split('-')[0] for n in names}),'classes_present':sorted(labels),'missing_mapping_ids':sorted(set(range(52))-labels)}
    summary[name]=detail
(ROOT/'reports/split_candidates.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:{p:{f:v[f] for f in ['sequences','recordings','missing_mapping_ids']} for p,v in parts.items()} for k,parts in summary.items()},indent=2))

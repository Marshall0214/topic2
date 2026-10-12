"""Measure development-set support for a single-variable 24 -> 128 control."""
from collections import Counter
import json
from pathlib import Path
import random
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import DATA,segments
run=json.loads((ROOT/'runs/small24_val8_e50_seed17/config.json').read_text())
split=json.loads((ROOT/run['split']).read_text())['splits']
pool=sorted(split['train'])
random.Random(run['seed']).shuffle(pool)
assert pool[:24]==run['selected']['train']
result={'test_used':False,'validation_unchanged':run['selected']['val'],'controls':{}}
for limit in [24,128]:
    names=pool[:limit]
    counts=Counter();bodies=[];views=Counter();missing=foreground=0
    for n in names:
        y=np.load(DATA/'gt_arr'/n).astype(np.int64)
        p=np.load(DATA/'features'/n).reshape(-1,2,25,3)
        counts.update(segments(y)[0]);views[n[-5]]+=1
        if np.any(p[:,1]):bodies.append(n)
        valid=np.any(p!=0,axis=(1,2,3))
        missing+=int(((~valid)&(y!=0)).sum());foreground+=int((y!=0).sum())
    result['controls'][str(limit)]={'names':names,'unique_recordings':len({n.split('-')[0] for n in names}),
        'views':dict(views),'body2_present_sequences':bodies,'action_segment_counts':dict(sorted(counts.items())),
        'action_segments':sum(counts.values()),'median_segments_per_action':float(np.median(list(counts.values()))),
        'classes_at_most_two_segments':sorted(k for k,v in counts.items() if v<=2),'foreground_missing_fraction':missing/max(1,foreground)}
(ROOT/'reports/failure_analysis50/data_control.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:{f:v[f] for f in ['unique_recordings','views','action_segments','median_segments_per_action','classes_at_most_two_segments','foreground_missing_fraction']}|{'body2_present_count':len(v['body2_present_sequences'])} for k,v in result['controls'].items()},indent=2))

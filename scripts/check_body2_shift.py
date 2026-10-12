"""Inference-only counterfactual; not a proposed correction or new benchmark."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import DATA,Meter,build_model,load_sequence,predict_logits
run=ROOT/'runs/small24_val8_e50_seed17'
config=json.loads((run/'config.json').read_text())
torch.set_num_threads(4)
device='cuda' if torch.cuda.is_available() else 'cpu'
names=config['selected']['val']
result={}
for kind in ['mstcn','protas']:
    model=build_model(kind,run/'train_graph.pkl').to(device).eval()
    model.load_state_dict(torch.load(run/kind/'best.pt',map_location=device,weights_only=True)['model'])
    meters={mode:{part:Meter() for part in ['all','body2_present','body2_absent']} for mode in ['original','mask_body2']}
    changed=total=0
    for name in names:
        feat,y=load_sequence(name)
        raw=np.load(DATA/'features'/name).reshape(-1,2,25,3)
        group='body2_present' if np.any(raw[:,1]) else 'body2_absent'
        alternatives={'original':feat,'mask_body2':feat.copy()}
        alternatives['mask_body2'][75:150]=0
        alternatives['mask_body2'][225:300]=0
        predictions={}
        with torch.inference_mode():
            for mode,value in alternatives.items():
                x=torch.from_numpy(value)[None].to(device)
                pred=predict_logits(model,kind,x,torch.ones(1,1,len(y),device=device)).argmax(1)[0].cpu().numpy()
                predictions[mode]=pred
                meters[mode]['all'].update(pred,y)
                meters[mode][group].update(pred,y)
        changed+=int((predictions['original']!=predictions['mask_body2']).sum());total+=len(y)
    result[kind]={'changed_frame_fraction':changed/total,'scores':{mode:{group:m.scores() for group,m in groups.items()} for mode,groups in meters.items()}}
output=ROOT/'reports/failure_analysis50/body2_counterfactual.json'
output.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

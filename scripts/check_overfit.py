"""Training-sequence memorization diagnostic, never a validation benchmark."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import load_sequence,progress_targets,make_graph,build_model,training_loss,predict_logits,Meter
torch.set_num_threads(4)
config=json.loads((ROOT/'runs/small24_val8_e10_seed17/config.json').read_text())
name=config['selected']['train'][0]
features,labels=load_sequence(name)
graph_path=ROOT/'reports/overfit_train_graph.pkl'
make_graph([labels],graph_path)
device='cuda' if torch.cuda.is_available() else 'cpu'
x=torch.from_numpy(features)[None].to(device)
y=torch.from_numpy(labels)[None].to(device)
progress=torch.from_numpy(progress_targets(labels))[None].to(device)
mask=torch.ones(1,1,len(labels),device=device)
report={'sequence':name,'training_only':True,'steps_limit':400,'models':{}}
for kind in ['mstcn','protas']:
    torch.manual_seed(17)
    model=build_model(kind,graph_path).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.0005)
    history=[]
    for step in range(1,401):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=training_loss(model,kind,x,y,progress,mask)
        loss.backward()
        optimizer.step()
        if step%50==0 or step==1:
            model.eval()
            with torch.inference_mode():
                pred=predict_logits(model,kind,x,mask).argmax(1)[0].cpu().numpy()
            meter=Meter()
            meter.update(pred,labels)
            row=dict(step=step,loss=float(loss.detach()),predicted_background_fraction=float((pred==0).mean()),**meter.scores())
            history.append(row)
            print(f'{kind} step={step} loss={row["loss"]:.3f} train_acc={row["accuracy"]:.2f} train_f1={row["f1_50"]:.2f} predicted_bg={row["predicted_background_fraction"]:.3f}',flush=True)
            if row['accuracy']>95 and row['f1_50']>90:
                break
    report['models'][kind]=history
    (ROOT/'reports/overfit_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    del model,optimizer
    if device=='cuda':torch.cuda.empty_cache()

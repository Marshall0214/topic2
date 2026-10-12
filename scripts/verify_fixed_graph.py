"""Verify frozen graph, retained graph-to-network gradients, and causality."""
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import build_model,make_graph,predict_logits,training_loss

torch.set_num_threads(4)
target=ROOT/'reports/protas_fixed_graph'
target.mkdir(exist_ok=True)
graph_path=target/'fixture_graph.pkl'
initial=make_graph([np.array([0,1,1,2,2,3]),np.array([1,3,3,0])],graph_path)
device='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(17)
model=build_model('protas',graph_path,graph_reference='official',task_graph='fixed').to(device)
assert not model.graph_learner.learnable
assert not model.graph_learner.matrix_pre.requires_grad and not model.graph_learner.matrix_suc.requires_grad
x=torch.randn(1,300,41,device=device);mask=torch.ones(1,1,41,device=device)
y=torch.randint(0,4,(1,41),device=device);p=torch.rand(1,52,41,device=device)
# Graph loss must remain differentiable to class/progress predictions.
cls=torch.randn(1,52,41,device=device,requires_grad=True)
progress=torch.full_like(cls,.5,requires_grad=True)
graph_loss=model.graph_learner(cls,progress)
graph_loss.backward()
cls_grad=float(cls.grad.abs().sum());progress_grad=float(progress.grad.abs().sum())
assert cls_grad>0 and progress_grad>0
before={k:v.clone() for k,v in model.state_dict().items()}
opt=torch.optim.Adam(model.parameters(),lr=.0005)
model.train();loss=training_loss(model,'protas',x,y,p,mask)
assert torch.isfinite(loss)
loss.backward();opt.step()
for key,value in initial.items():
    assert torch.equal(getattr(model.graph_learner,key).detach(),torch.as_tensor(value,device=device))
assert not torch.equal(before['stage1.conv_1x1.weight'],model.state_dict()['stage1.conv_1x1.weight'])
model.eval()
with torch.inference_mode():
    full=predict_logits(model,'protas',x,mask)
    alt=x.clone();alt[...,21:]*=100
    future_error=float((predict_logits(model,'protas',alt,mask)[...,:21]-full[...,:21]).abs().max())
    prefix_error=float((predict_logits(model,'protas',x[...,:21],mask[...,:21])-full[...,:21]).abs().max())
assert max(future_error,prefix_error)<1e-5
result=dict(device=device,graph_frozen=True,graph_exact_initial_after_adam=True,network_updated=True,
            graph_loss_class_gradient_l1=cls_grad,graph_loss_progress_gradient_l1=progress_grad,
            future_error=future_error,prefix_error=prefix_error,passed=True)
(target/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

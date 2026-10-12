"""Check adapted single-sequence loss against pinned official equations."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from baseline import PROTAS, build_model, training_loss, make_graph

torch.set_num_threads(4)
out = ROOT/'reports/protas_official_audit'
out.mkdir(exist_ok=True)
graph_path = out/'fixture_graph.pkl'
make_graph([np.array([0,1,1,2,0,3])], graph_path)
torch.manual_seed(19)
official = PROTAS.MultiStageModel(4,10,64,300,52,causal=True,use_graph=True,
                                init_graph_path=str(graph_path),learnable=True)
torch.manual_seed(19)
local = build_model('protas', graph_path)
assert all(torch.equal(value, local.state_dict()[key]) for key,value in official.state_dict().items())
x = torch.randn(1,300,35)
y = torch.randint(0,52,(1,35))
prg = torch.rand(1,52,35)
mask = torch.ones(1,52,35)
official.train(); local.train()
torch.manual_seed(37)
pred,progress = official(x,mask)
loss = torch.zeros(())
for p, q in zip(pred,progress):
    loss = loss+F.cross_entropy(p.transpose(2,1).contiguous().view(-1,52),y.view(-1))
    loss = loss+.15*((F.log_softmax(p[:,:,1:],1)-F.log_softmax(p.detach()[:,:,:-1],1)).square().clamp(0,16)*mask[:,:,1:]).mean()
    loss = loss+F.mse_loss(q,prg)
loss = loss+.1*official.graph_learner(pred[-1],progress[-1])
torch.manual_seed(37)
local_loss = training_loss(local,'protas',x,y,prg,mask[:,:1])
loss.backward(); local_loss.backward()
gradient_error = max(float((p.grad-dict(local.named_parameters())[name].grad).abs().max())
                     for name,p in official.named_parameters() if p.grad is not None)
loss_error = float((loss-local_loss).detach().abs())
official.eval();local.eval()
with torch.inference_mode():
    po,qo=official(x,mask);pl,ql=local(x,mask[:,:1])
    output_error=float((po-pl).abs().max())
    inference_error=float((official.graph_learner.inference(po[-1],qo[-1])-local.graph_learner.inference(pl[-1],ql[-1])).abs().max())
    for m in [official,local]:m.graph_learner.matrix_pre.add_(.5)
    official_reg=float((official.graph_learner.matrix_pre-official.graph_learner.matrix_pre_original).square().mean())
    local_reg=float((local.graph_learner.matrix_pre-local.graph_learner.matrix_pre_original).square().mean())
checks=[]
for source in json.loads((ROOT/'sources/manifest.json').read_text())[0]['files']:
    checks.append(dict(path=source['path'],matches_pinned_manifest=hashlib.sha256((ROOT/'sources/protas/upstream'/source['path']).read_bytes()).hexdigest()==source['sha256']))
assert loss_error<1e-5 and gradient_error<1e-5 and output_error==0 and inference_error==0
assert official_reg==0 and abs(local_reg-.25)<1e-6
result=dict(scope='Synthetic batch1 full-sequence parity at identical initialization; not full training equivalence',
    official_configuration='causal=True,use_graph=True,learnable=True,input_dim=300,classes=52',
    loss_error=loss_error,gradient_error=gradient_error,output_error=output_error,inference_error=inference_error,
    after_equal_parameter_shift=dict(official_reference_regularizer=official_reg,local_reference_regularizer=local_reg),
    official_reference_alias=official.graph_learner.matrix_pre.data_ptr()==official.graph_learner.matrix_pre_original.data_ptr(),
    local_reference_alias=local.graph_learner.matrix_pre.data_ptr()==local.graph_learner.matrix_pre_original.data_ptr(),
    pinned_manifest_checks=checks)
(out/'parity_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

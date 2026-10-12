"""Check adapted single-sequence loss against pinned official equations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from baseline import PROTAS, OfficialMeter, build_model, training_loss, make_graph

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
spec=importlib.util.spec_from_file_location('official_eval',ROOT/'sources/protas/upstream/eval.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
# Compatibility adapter confined to imported evaluator; upstream stays untouched.
class NumpyCompat:
    def __getattr__(self,name):
        return float if name=='float' else getattr(np,name)
ev.np=NumpyCompat()
rng=np.random.default_rng(41)
meter=OfficialMeter(); expected_edit=[]; tp=np.zeros(3);fp=np.zeros(3);fn=np.zeros(3)
for _ in range(16):
    pred=np.repeat(rng.integers(0,4,14),3); gt=np.repeat(rng.integers(0,4,14),3)
    meter.update(pred,gt)
    ps=[str(v) for v in pred];ys=[str(v) for v in gt]
    expected_edit.append(ev.edit_score(ps,ys,bg_class=['0']))
    for i,threshold in enumerate([.1,.25,.5]):
        a,b,c=ev.f_score(ps,ys,threshold,bg_class=['0']);tp[i]+=a;fp[i]+=b;fn[i]+=c
assert abs(meter.scores()['edit']-np.mean(expected_edit))<1e-8
assert np.allclose([meter.scores()[f'f1_{i}'] for i in [10,25,50]],100*2*tp/(2*tp+fp+fn))
example=OfficialMeter();example.update([1,0,1],[1,1,1])
assert example.scores()['edit']==100 and abs(example.scores()['foreground_accuracy']-200/3)<1e-8
result['official_metric_parity']=True
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(19);a=build_model('protas',graph_path,graph_reference='official').cuda()
    torch.manual_seed(19);b=build_model('protas',graph_path).cuda()
    gpu_alias=a.graph_learner.matrix_pre.data_ptr()==a.graph_learner.matrix_pre_original.data_ptr()
    oa=torch.optim.Adam(a.parameters(),lr=.0005);ob=torch.optim.Adam(b.parameters(),lr=.0005)
    xa,ya,pa,ma=x.cuda(),y.cuda(),prg.cuda(),mask[:,:1].cuda()
    torch.manual_seed(37);la=training_loss(a,'protas',xa,ya,pa,ma);la.backward()
    torch.manual_seed(37);lb=training_loss(b,'protas',xa,ya,pa,ma);lb.backward()
    gpu_gradient_error=max(float((p.grad-dict(b.named_parameters())[key].grad).detach().abs().max())
                           for key,p in a.named_parameters() if p.grad is not None)
    oa.step();ob.step()
    state_error=max(float((value-b.state_dict()[key]).detach().abs().max()) for key,value in a.state_dict().items())
    assert not gpu_alias and gpu_gradient_error<1e-5 and state_error<5e-5
    result['gpu_official_vs_copy']=dict(official_reference_alias_after_cuda=gpu_alias,
        one_step_loss_error=float((la-lb).detach().abs()),gradient_max_error=gpu_gradient_error,one_step_state_max_error=state_error,
        note='CPU shared storage separates on CUDA conversion in this runtime; CPU-only alias conclusion does not apply to completed GPU training')
(out/'parity_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

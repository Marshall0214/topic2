"""Verify spatial layout, temporal initialization, causality and gradients."""
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import build_model
from spatial_encoders import build_spatial_model,unpack_features

torch.set_num_threads(4)
torch.manual_seed(17)
original=build_model('mstcn')
x=torch.randn(1,300,81)
mask=torch.ones(1,1,81)
probe=torch.arange(300.).reshape(1,300,1)
nodes=unpack_features(probe)
assert nodes.shape==(2,6,25,1)
assert nodes[1,0,0,0]==75 and nodes[0,3,0,0]==150 and nodes[1,5,24,0]==299
results={}
for name in ['linear','mlp','gcn']:
    torch.manual_seed(17)
    model=build_spatial_model(name).eval()
    for key,value in model.state_dict().items():
        if not key.startswith('stage1.conv_1x1.'):
            assert torch.equal(value,original.state_dict()[key]),key
    with torch.no_grad():
        y=model(x,mask)[0]
        changed=x.clone();changed[:,:,40:]+=torch.randn_like(changed[:,:,40:])*10
        future=float((y[:,:,:,:40]-model(changed,mask)[0][:,:,:,:40]).abs().max())
        prefix=float((y[:,:,:,:40]-model(x[:,:,:40],mask[:,:,:40])[0]).abs().max())
        assert future<1e-6 and prefix<1e-5
        if name=='linear':
            assert torch.equal(y,original.eval()(x,mask)[0])
        if name=='gcn':
            body_features=unpack_features(torch.zeros_like(x))
            assert (body_features==0).all()
    model.train();model.zero_grad()
    loss=model(x,mask)[0].square().mean();loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    results[name]=dict(parameters=sum(p.numel() for p in model.parameters()),
                       encoder_parameters=sum(p.numel() for p in model.stage1.conv_1x1.parameters()),
                       future_difference=future,prefix_difference=prefix)
assert abs(results['gcn']['encoder_parameters']-results['mlp']['encoder_parameters'])/results['gcn']['encoder_parameters']<.01
target=ROOT/'reports/spatial_pilot';target.mkdir(exist_ok=True)
(target/'verification.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))

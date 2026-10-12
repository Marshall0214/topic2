"""Meaningful input, scoring, graph provenance and adapter causality checks."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import Meter,build_model,edit_score,make_graph,predict_logits,progress_targets,segments,skeleton_features

torch.set_num_threads(4)
torch.manual_seed(17)
out = ROOT/'reports'
out.mkdir(exist_ok=True)
raw = np.arange(8*150,dtype=np.float32).reshape(8,150)+1
raw[2,75:] = 0
feature = skeleton_features(raw)
assert feature.shape == (300,8)
assert not feature[75:150,2].any() and not feature[225:300,2:4].any()
changed = raw.copy()
changed[4:] *= 5
assert np.array_equal(skeleton_features(changed)[:,:4],feature[:,:4])
assert not feature[150:,0].any()
assert segments([1]) == ([1],[0],[1])
meter = Meter()
meter.update([1],[1])
assert meter.scores()['f1_50'] == 100
meter = Meter()
meter.update([0,0],[0,0])
assert meter.scores()['edit']==100 and meter.scores()['f1_50']==0
meter = Meter()
meter.update([1,0,1],[1,1,1])
assert meter.scores()['edit']==50
meter = Meter()
meter.update([1,1],[0,0])
assert meter.scores()['f1_50']==0
meter = Meter()
meter.update([0,0],[1,1])
assert meter.scores()['f1_50']==0
target = progress_targets(np.array([0,1,1,0,2]))
assert np.allclose(target[1],[0,.5,1,0,0]) and not target[0].any()
graph_path = out/'shared_check_graph.pkl'
make_graph([np.array([0,1,1,2,2]),np.array([2,3,0])],graph_path)

# Check the legacy compatibility path against the actual existing evaluator.
path = Path(r'D:\AQA\research\keraal_literature\upstream\dest\libs\metric.py')
spec = importlib.util.spec_from_file_location('legacy_metric',path)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
mapping = {0:'NONE',1:'A',2:'B',3:'C'}
reference,local = legacy.ScoreMeter(mapping),Meter(legacy=True)
rng = np.random.default_rng(17)
for _ in range(8):
    pred = np.repeat(rng.integers(0,4,8),3)
    gt = np.repeat(rng.integers(0,4,8),3)
    reference.update(pred[None],gt[None])
    local.update(pred,gt)
acc,edit,f1 = reference.get_scores()
ours = local.scores()
assert np.allclose([acc,edit,*f1],[ours[k] for k in ['accuracy','edit','f1_10','f1_25','f1_50']],atol=1e-5)

errors = {}
for kind in ['mstcn','protas']:
    model = build_model(kind,graph_path).eval()
    x = torch.randn(1,300,33)
    def run(value):
        return predict_logits(model,kind,value,torch.ones(1,1,value.shape[-1]))
    with torch.inference_mode():
        full = run(x)
        alt = x.clone()
        alt[...,17:] = torch.randn_like(alt[...,17:])*50
        future_error = (run(alt)[...,:17]-full[...,:17]).abs().max().item()
        prefix_error = max((run(x[...,:i+1])[...,-1]-full[...,i]).abs().max().item() for i in [0,1,7,16,32])
        run(torch.randn_like(x))
        reset_error = (run(x)-full).abs().max().item()
    assert max(future_error,prefix_error,reset_error)<1e-5
    if kind=='protas':
        graph = model.graph_learner
        assert graph.matrix_pre.data_ptr()!=graph.matrix_pre_original.data_ptr()
        assert 'matrix_pre_original' in dict(graph.named_buffers())
    errors[kind]=dict(future_error=future_error,prefix_error=prefix_error,state_isolation_error=reset_error,parameters=sum(p.numel() for p in model.parameters()))
result = dict(passed=True,input_checks=True,canonical_edge_cases=True,legacy_parity=True,progress_and_training_graph_checks=True,model_errors=errors,
              scope='Untrained models with nonzero synthetic training graph; no task accuracy claim; prefix replay, not cached streaming')
(out/'shared_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))

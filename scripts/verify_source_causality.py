"""Synthetic checks on untouched upstream models; no training or dataset scores."""
import importlib.util
import json
import pickle
from pathlib import Path
import contextlib
import types
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(4)
torch.manual_seed(17)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

protas = load('audit_protas', ROOT/'sources/protas/upstream/model.py')
mstcn = types.ModuleType('audit_mstcn')
# Upstream Trainer contains Python-2 print syntax. Execute only unchanged model
# definitions before Trainer; do not rewrite the archived upstream source.
mstcn_path = ROOT/'sources/mstcn/upstream/model.py'
exec(compile(mstcn_path.read_text().split('class Trainer:')[0],str(mstcn_path),'exec'),mstcn.__dict__)
results = {'device': 'cpu', 'torch': str(torch.__version__), 'tolerance': 1e-5, 'training': False}
(ROOT/'reports').mkdir(exist_ok=True)
with contextlib.nullcontext(ROOT/'reports') as temp:
    graph_path = Path(temp)/'synthetic_graph.pkl'
    with graph_path.open('wb') as f:
        pickle.dump({'matrix_pre': np.zeros((52,52),dtype=np.float32), 'matrix_suc': np.zeros((52,52),dtype=np.float32)},f)
    model = protas.MultiStageModel(4,10,64,300,52,causal=True,use_graph=True,init_graph_path=str(graph_path),learnable=False).eval()
    x = torch.randn(1,300,33)
    def run(value):
        mask = torch.ones(1,1,value.shape[-1])
        out, progress = model(value, mask)
        return model.graph_learner.inference(out[-1],progress[-1])
    with torch.inference_mode():
        full = run(x)
        perturbed = x.clone()
        perturbed[...,17:] = torch.randn_like(perturbed[...,17:])*100
        future_error = (run(perturbed)[...,:17]-full[...,:17]).abs().max().item()
        prefix_errors = {str(t): (run(x[...,:t+1])[...,-1]-full[...,t]).abs().max().item() for t in [0,1,7,16,32]}
        repeat_error = (run(x)-full).abs().max().item()
        offline = mstcn.MultiStageModel(4,10,64,300,52).eval()
        ofull = offline(x,torch.ones(1,1,33))[-1]
        oalt = offline(perturbed,torch.ones(1,1,33))[-1]
        offline_future_change = (ofull[...,:17]-oalt[...,:17]).abs().max().item()
    results['protas'] = dict(configuration='official 4 stages,10 layers,64 channels;dim=300,52 classes; synthetic zero graph',parameters=sum(p.numel() for p in model.parameters()),output_shape=list(full.shape),future_perturbation_error=future_error,prefix_errors=prefix_errors,repeat_error=repeat_error,passes=future_error<1e-5 and max(prefix_errors.values())<1e-5 and repeat_error<1e-5)
    results['mstcn_offline_control'] = dict(future_perturbation_change=offline_future_change,detected_future_dependency=offline_future_change>1e-5,loader='Only unchanged model definitions before Python-2 Trainer were executed.')
    results['limitations'] = ['Untrained synthetic inputs; no accuracy conclusions.', 'Zero graph tests model and cumulative graph inference shape only; learned nonzero graph still requires checking.', 'Prefix replay is not cached streaming inference and cannot establish constant per-frame cost.', 'Not a reproduction of DPTM skeleton input preprocessing.']
assert results['protas']['passes'] and results['mstcn_offline_control']['detected_future_dependency']
(ROOT/'reports/source_causality.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print(json.dumps(results,indent=2))

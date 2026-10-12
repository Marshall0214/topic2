"""Run three frozen validation-only spatial controls sequentially."""
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
cfg=json.loads((ROOT/'configs/spatial_pilot.json').read_text())
report=ROOT/'reports/spatial_pilot'
report.mkdir(exist_ok=True)
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():
    raise RuntimeError('Commit experiment code before launching the frozen pilot')
for encoder in cfg['encoders']:
    run=cfg['run_prefix']+'_'+encoder
    if (ROOT/'runs'/run).exists():
        raise FileExistsError(f'{run} already exists; do not overwrite')
for encoder in cfg['encoders']:
    run=cfg['run_prefix']+'_'+encoder
    command=[sys.executable,'-B',str(ROOT/'scripts/train_small.py'),'--model','mstcn',
             '--run',run,'--spatial-encoder',encoder,'--device','cuda','--clip','0']
    for k in ['split','train_limit','val_limit','epochs','seed','stages','layers','width','lr','normalization']:
        command.extend(['--'+k.replace('_','-'),str(cfg[k])])
    print('Starting',run,flush=True)
    with (report/(encoder+'.log')).open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    (ROOT/'runs'/run/'experiment_provenance.json').write_text(json.dumps(dict(git_commit=revision,protocol=cfg,command=command),indent=2),encoding='utf-8')
    print('Completed',run,flush=True)
print('Pilot complete. No test-set evaluation was performed.',flush=True)

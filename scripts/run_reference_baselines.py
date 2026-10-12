"""Frozen full-data baseline launcher. Full training is explicitly invoked."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--run',default='dptm_reference_pku_dev_seed17')
parser.add_argument('--model',choices=['mstcn','protas','both'],default='both')
parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
parser.add_argument('--dry-run',action='store_true')
args=parser.parse_args()
cfg=json.loads((ROOT/'configs/dptm_reference_pku.json').read_text())
command=[sys.executable,'-B',str(ROOT/'scripts/train_small.py'),'--model',args.model,'--run',args.run,'--device',args.device]
for key in ['split','train_limit','val_limit','epochs','clip','seed','stages','layers','width','lr','normalization']:
    command.extend(['--'+key.replace('_','-'),str(cfg[key])])
if cfg.get('trainval'):command.append('--trainval')
print('Frozen protocol:',json.dumps(cfg,indent=2),flush=True)
print('Command:',subprocess.list2cmdline(command),flush=True)
if not args.dry_run:
    subprocess.run(command,cwd=ROOT,check=True)
    folder=ROOT/'runs'/args.run
    (folder/'reference_protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')

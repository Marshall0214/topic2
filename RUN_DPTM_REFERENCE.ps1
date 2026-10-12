$ErrorActionPreference = 'Stop'
$py = 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe'
Set-Location $PSScriptRoot
& $py -B .\scripts\verify_shared.py
if ($LASTEXITCODE -ne 0) { throw 'Baseline verification failed.' }
& $py -B .\scripts\run_reference_baselines.py --run dptm_reference_pku_dev_seed17 --model both --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Training failed; keep checkpoints and logs.' }
& $py -B .\scripts\evaluate_run.py --run dptm_reference_pku_dev_seed17 --model both --checkpoint best --split test --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed; keep logs.' }

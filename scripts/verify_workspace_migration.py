"""Migration acceptance: checkpoint reload, short validation inference, no training."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from baseline import DATA,build_model,load_sequence,predict_logits

def main():
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    result={'device':device,'scope':'One validation prefix (96 frames) per model; no training, no test evaluation','models':{}}
    split=ROOT/'configs/dest_compat.json'
    for kind,name in [('mstcn','mstcn_pku_dev_e100_resume_seed17'),('protas','protas_pku_fixedgraph_e100_resume_seed17')]:
        run=ROOT/'runs'/name
        cfg=json.loads((run/'config.json').read_text())
        assert hashlib.sha256(split.read_bytes()).hexdigest()==cfg['split_sha256']
        model=build_model(kind,run/'train_graph.pkl',cfg['stages'],cfg['layers'],cfg['width'],graph_reference=cfg.get('graph_reference','independent_copy'),task_graph=cfg.get('task_graph','learnable')).to(device).eval()
        ck=torch.load(run/kind/'best.pt',map_location=device,weights_only=True)
        model.load_state_dict(ck['model'],strict=True)
        x,y=load_sequence(cfg['selected']['val'][0])
        x=torch.from_numpy(x[:,:96]).unsqueeze(0).to(device)
        with torch.inference_mode(): out=predict_logits(model,kind,x,torch.ones(1,1,x.shape[-1],device=device))
        assert torch.isfinite(out).all()
        command=[sys.executable,'-B',str(ROOT/'scripts/train_small.py'),'--model',kind,'--run','migration_resume_check_unused_'+kind,'--split','configs/dest_compat.json','--epochs','101','--train-limit','0','--val-limit','0','--resume-from',name,'--check-resume','--seed',str(cfg['seed']),'--stages',str(cfg['stages']),'--layers',str(cfg['layers']),'--width',str(cfg['width']),'--lr',str(cfg['lr']),'--normalization',cfg['normalization'],'--clip',str(cfg['clip']),'--graph-reference',cfg.get('graph_reference','independent_copy'),'--task-graph',cfg.get('task_graph','learnable')]
        check=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
        # Preserve strict resume guards; report historical source drift rather than bypass it.
        resume_passed=check.returncode==0
        assert not (ROOT/'runs'/('migration_resume_check_unused_'+kind)).exists()
        result['models'][kind]={'best_epoch':ck['epoch'],'logits_shape':list(out.shape),'finite':True,'split_hash_matches':True,'resume_passed':resume_passed,'resume_check':(check.stdout+check.stderr).strip()}
        del model,ck,x,out
    from verify_specscalpel_training import build_model as build_spec,ActionSegmentationDataset,ToTensor,TempDownSamp,collate_fn
    from torchvision.transforms import Compose
    run=ROOT/'runs/specscalpel_full_seed17'
    cfg=json.loads((run/'config.json').read_text())
    assert hashlib.sha256(split.read_bytes()).hexdigest()==cfg['split_sha256']
    model=build_spec().to(device).eval()
    best=torch.load(run/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(best['model'],strict=True)
    dataset=ActionSegmentationDataset('PKU-subject',transform=Compose([ToTensor(),TempDownSamp(1)]),mode='validation',dataset_dir=str(DATA.parent),csv_dir=str(run/'csv'),augmentations=False)
    sample=collate_fn([dataset[0]])
    x=sample['feature'][...,:96].to(device); mask=sample['mask'][...,:96].to(device)
    joints=torch.from_numpy(np.load(ROOT/'sources/specscalpel/assets/PKU-subject_joints_pool.npy')).to(device)
    distance=((joints[:,None]-joints[None])**2).sum(-1).sqrt()
    graph=1-(distance-distance.min())/(distance.max()-distance.min())
    with torch.inference_mode(): logits,bounds=model(x,mask,graph)
    assert torch.isfinite(logits).all() and torch.isfinite(bounds).all()
    last=torch.load(run/'last.pt',map_location='cpu',weights_only=True)
    assert last['selected']==cfg['selected']
    assert last['history'][-1]['epoch']==last['epoch']
    assert {'python','numpy','torch'}<=set(last['rng'])
    model.load_state_dict(last['model'],strict=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    optimizer.load_state_dict(last['optimizer'])
    result['models']['specscalpel']={'best_epoch':best['epoch'],'last_epoch':last['epoch'],'logits_shape':list(logits.shape),'finite':True,'split_hash_matches':True,'last_model_optimizer_history_rng_loadable':True,'resume_scope':'State restoration only, no optimizer step or exact RNG continuation experiment'}
    result['checkpoint_and_inference_passed']=True
    result['all_resume_checks_passed']=all(m.get('resume_passed',True) for m in result['models'].values())
    (ROOT/'reports/workspace_migration/verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

"""Shared small-run entry. Validation only; no automatic test-set evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baseline import DATA, Meter, OfficialMeter, build_model, load_sequence, make_graph, predict_logits, progress_targets, training_loss
from normalization import TrainingStandardizer, feature_validity
from spatial_encoders import build_spatial_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=['mstcn','protas','both'],default='both')
    parser.add_argument('--run',required=True,help='New directory name; existing runs are never overwritten')
    parser.add_argument('--split',default='configs/recording_holdout_seed17.json')
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--train-limit',type=int,default=24,help='0 selects entire training split')
    parser.add_argument('--val-limit',type=int,default=8,help='0 selects entire validation split')
    parser.add_argument('--clip',type=int,default=0,help='0=full sequence; positive crops are smoke checks only')
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--stages',type=int,default=4)
    parser.add_argument('--layers',type=int,default=10)
    parser.add_argument('--width',type=int,default=64)
    parser.add_argument('--lr',type=float,default=.0005)
    parser.add_argument('--normalization',choices=['none','train-standard'],default='none')
    parser.add_argument('--trainval',action='store_true',help='Final fixed-epoch training on train+val; no validation or best checkpoint')
    parser.add_argument('--resume-from',help='Existing run to continue from last.pt into a NEW run directory')
    parser.add_argument('--check-resume',action='store_true',help='Validate resume compatibility only; do not train or create a run')
    parser.add_argument('--spatial-encoder',choices=['linear','mlp','gcn'],default='linear')
    parser.add_argument('--graph-reference',choices=['independent_copy','official'],default='independent_copy')
    args = parser.parse_args()
    if args.spatial_encoder != 'linear' and args.model != 'mstcn':
        parser.error('Experimental spatial encoders support MS-TCN only')
    if Path(args.run).name != args.run or args.run in ['.','..']:
        parser.error('--run must be a directory name')
    if args.epochs<1 or min(args.train_limit,args.val_limit,args.clip)<0:
        parser.error('Invalid epochs or limits')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU fallback')
    split_path = Path(args.split)
    if not split_path.is_absolute():
        split_path = ROOT/split_path
    split = json.loads(split_path.read_text())['splits']
    assert not(set(split['train']) & set(split['val']))
    assert not((set(split['train'])|set(split['val'])) & set(split['test']))
    selected = {}
    for part,limit in [('train',args.train_limit),('val',args.val_limit)]:
        names = sorted(split[part])
        random.Random(args.seed+(0 if part=='train' else 1)).shuffle(names)
        selected[part] = names[:limit or None]
    if args.trainval:
        if args.train_limit or args.val_limit:
            parser.error('--trainval requires --train-limit 0 --val-limit 0')
        names = sorted(set(split['train'])|set(split['val']))
        random.Random(args.seed).shuffle(names)
        selected = {'train':names,'val':[]}
    resume_run = None
    if args.check_resume and not args.resume_from:
        parser.error('--check-resume requires --resume-from')
    if args.resume_from:
        if Path(args.resume_from).name != args.resume_from or args.resume_from in ['.','..']:
            parser.error('--resume-from must be a run directory name')
        if args.clip or args.trainval or args.model == 'both':
            parser.error('Resume currently supports one model, full sequences, and separate validation only')
        resume_run = ROOT/'runs'/args.resume_from
        old = json.loads((resume_run/'config.json').read_text())
        if old.get('graph_reference','independent_copy') != args.graph_reference:
            raise ValueError('Cannot resume with different graph reference behavior')
        if old.get('spatial_encoder','linear') != args.spatial_encoder:
            raise ValueError('Resume spatial encoder mismatch')
        for key in ['seed','stages','layers','width','lr','normalization','clip','trainval','batch_size']:
            actual = 1 if key == 'batch_size' else getattr(args,key)
            if old[key] != actual:
                raise ValueError(f'Resume configuration mismatch: {key}')
        if old['selected'] != selected or old['split_sha256'] != hashlib.sha256(split_path.read_bytes()).hexdigest():
            raise ValueError('Resume split or selected sequence order changed')
        for source in [ROOT/'src/baseline.py',ROOT/'src/normalization.py',ROOT/'sources/protas/upstream/model.py']:
            if hashlib.sha256(source.read_bytes()).hexdigest() != old['source_hashes'][str(source.relative_to(ROOT))]:
                raise ValueError(f'Model/input source changed: {source}')
        if args.spatial_encoder != 'linear':
            source=ROOT/'src/spatial_encoders.py'
            if hashlib.sha256(source.read_bytes()).hexdigest()!=old['source_hashes'][str(source.relative_to(ROOT))]:
                raise ValueError('Spatial encoder source changed')
        previous = torch.load(resume_run/args.model/'last.pt',map_location='cpu',weights_only=True)
        if previous['kind'] != args.model or 'optimizer' not in previous:
            raise ValueError('Missing or incompatible optimizer/model checkpoint')
        if args.epochs <= previous['epoch']:
            raise ValueError('--epochs is the TOTAL target epoch count; must exceed saved epoch')
        saved_history = json.loads((resume_run/args.model/'history.json').read_text())
        if [r['epoch'] for r in saved_history] != list(range(1,previous['epoch']+1)):
            raise ValueError('Incomplete history or history/checkpoint mismatch')
        print(f'Resume verified: {args.model} epoch {previous["epoch"]} -> {args.epochs}; model AND Adam state restored; old run preserved.',flush=True)
        del previous
        if args.check_resume:
            return
    run = ROOT/'runs'/args.run
    run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cache = {n:load_sequence(n) for n in selected['train']+selected['val']}
    normalization_record = None
    if args.normalization=='train-standard':
        validity = {n:feature_validity(np.load(DATA/'features'/n,allow_pickle=False)) for n in cache}
        standardizer = TrainingStandardizer.fit((cache[n][0],validity[n]) for n in selected['train'])
        stats_path = run/'normalization.npz'
        standardizer.save(stats_path)
        cache = {n:(standardizer.transform(x,validity[n]),y) for n,(x,y) in cache.items()}
        normalization_record = dict(file='normalization.npz',sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest(),
                                    fitted_on=selected['train'],std_floor=1e-3,unobserved_channels=int((standardizer.count==0).sum()),
                                    validity='per-person current relative; adjacent-pair backward displacement; invalid stays zero',clipping=False)
    progress = {n:progress_targets(y) for n,(_,y) in cache.items()}
    graph_path = run/'train_graph.pkl'
    make_graph([cache[n][1] for n in selected['train']],graph_path)
    config = dict(vars(args),selected=selected,data_root=str(DATA),split_path=str(split_path),
                  split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),test_used=False,
                  features='per-person root-relative XYZ then backward XYZ;300 dims',
                  checkpoint_rule='fixed final epoch; no validation or best checkpoint' if args.trainval else 'max foreground_canonical (Edit+F1@50)/2 on selected validation',
                  background_id=0,classes=52,batch_size=1,optimizer='Adam;weight_decay=0',
                  torch=str(torch.__version__),gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                  normalization_record=normalization_record,
                  training_classes=sorted(set(np.concatenate([cache[n][1] for n in selected['train']]).tolist())),
                  source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/baseline.py',ROOT/'src/normalization.py',ROOT/'src/spatial_encoders.py',Path(__file__),ROOT/'sources/protas/upstream/model.py']})
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    snapshot = run/'source_snapshot'
    snapshot.mkdir()
    for source,target in [(ROOT/'src/baseline.py','baseline.py'),(ROOT/'src/normalization.py','normalization.py'),(ROOT/'src/spatial_encoders.py','spatial_encoders.py'),(Path(__file__),'train_small.py'),(ROOT/'sources/protas/upstream/model.py','pinned_protas_model.py'),(split_path,'split.json')]:
        shutil.copyfile(source,snapshot/target)
    print(f'device={args.device}; train={len(selected["train"])}; val={len(selected["val"])}; classes_present={len(config["training_classes"])}; test_used=False',flush=True)
    print(f'normalization={args.normalization}; statistics from selected training sequences only',flush=True)
    summaries = {}
    for kind in (['mstcn','protas'] if args.model=='both' else [args.model]):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        folder = run/kind
        folder.mkdir()
        model = (build_spatial_model(args.spatial_encoder,args.stages,args.layers,args.width) if kind=='mstcn' else build_model(kind,graph_path,args.stages,args.layers,args.width,graph_reference=args.graph_reference)).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        rng = np.random.default_rng(args.seed)
        history,best = [],-float('inf')
        start_epoch = 1
        if resume_run is not None:
            previous = torch.load(resume_run/kind/'last.pt',map_location=args.device,weights_only=True)
            model.load_state_dict(previous['model'],strict=True)
            optimizer.load_state_dict(previous['optimizer'])
            history = json.loads((resume_run/kind/'history.json').read_text())
            best = max((r['foreground_canonical']['edit']+r['foreground_canonical']['f1_50'])/2 for r in history)
            start_epoch = previous['epoch']+1
            # Legacy checkpoints lack NumPy generator state. Full-sequence training
            # consumes exactly one permutation per epoch, so reconstruct that state.
            for _ in range(previous['epoch']):
                rng.permutation(len(selected['train']))
            shutil.copyfile(resume_run/kind/'best.pt',folder/'best.pt')
            del previous
        for epoch in range(start_epoch,args.epochs+1):
            begin = time.perf_counter()
            torch.manual_seed(args.seed+epoch)
            model.train()
            losses = []
            for index in rng.permutation(len(selected['train'])):
                name = selected['train'][index]
                features,labels = cache[name]
                start = int(rng.integers(max(1,len(labels)-args.clip+1))) if args.clip else 0
                end = min(len(labels),start+args.clip) if args.clip else len(labels)
                x = torch.from_numpy(features[:,start:end].copy())[None].to(args.device)
                y = torch.from_numpy(labels[start:end].copy())[None].to(args.device)
                prg = torch.from_numpy(progress[name][:,start:end].copy())[None].to(args.device)
                mask = torch.ones(1,1,end-start,device=args.device)
                optimizer.zero_grad(set_to_none=True)
                loss = training_loss(model,kind,x,y,prg,mask)
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite loss: {kind}/{name}')
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise RuntimeError('Nonfinite gradient')
                optimizer.step()
                losses.append(float(loss.detach()))
            if args.trainval:
                record=dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-begin,
                            validation_performed=False)
                history.append(record)
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,kind=kind,config=config,validation=None)
                torch.save(state,folder/'last.pt')
                (folder/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
                print(f'{kind} final trainval epoch {epoch}/{args.epochs} loss={record["loss"]:.4f} seconds={record["seconds"]:.2f}; no validation',flush=True)
                continue
            model.eval()
            canonical,legacy = Meter(),Meter(legacy=True)
            official = OfficialMeter()
            first_stage = Meter()
            predicted_background = true_background = foreground_correct = foreground_frames = validation_frames = 0
            predicted_classes = set()
            with torch.inference_mode():
                for name in selected['val']:
                    features,labels = cache[name]
                    x = torch.from_numpy(features)[None].to(args.device)
                    mask = torch.ones(1,1,len(labels),device=args.device)
                    outputs,progress_outputs = model(x,mask)
                    logits = outputs[-1] if kind=='mstcn' else model.graph_learner.inference(outputs[-1],progress_outputs[-1])
                    pred = logits.argmax(1)[0].cpu().numpy()
                    canonical.update(pred,labels)
                    legacy.update(pred,labels)
                    official.update(pred,labels)
                    first_stage.update(outputs[0].argmax(1)[0].cpu().numpy(),labels)
                    predicted_background += int((pred==0).sum())
                    true_background += int((labels==0).sum())
                    foreground_correct += int(((pred==labels)&(labels!=0)).sum())
                    foreground_frames += int((labels!=0).sum())
                    validation_frames += len(labels)
                    predicted_classes.update(map(int,np.unique(pred)))
            scores = canonical.scores()
            record = dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-begin,
                          foreground_canonical=scores,dest_legacy=legacy.scores(),protas_official=official.scores())
            record['validation_diagnostics'] = dict(predicted_background_fraction=predicted_background/validation_frames,
                    true_background_fraction=true_background/validation_frames,foreground_accuracy=100*foreground_correct/max(1,foreground_frames),
                    predicted_classes=sorted(predicted_classes),first_stage_raw=first_stage.scores())
            history.append(record)
            score = (scores['edit']+scores['f1_50'])/2
            state = dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,kind=kind,config=config,validation=record)
            torch.save(state,folder/'last.pt')
            if score>best:
                best = score
                torch.save(state,folder/'best.pt')
            (folder/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(f'{kind} epoch {epoch}/{args.epochs} loss={record["loss"]:.4f} val Acc={scores["accuracy"]:.2f} Edit={scores["edit"]:.2f} F1@50={scores["f1_50"]:.2f} pred_bg={predicted_background/validation_frames:.1%} first_stage_F1={first_stage.scores()["f1_50"]:.2f} seconds={record["seconds"]:.2f}',flush=True)
            if predicted_background==validation_frames:
                print('WARNING: all validation predictions are background; Acc equals background fraction, not successful action recognition.',flush=True)
        summaries[kind] = dict(parameters=sum(p.numel() for p in model.parameters()),best_validation_score=None if args.trainval else best,
                               selected_epoch=args.epochs if args.trainval else max(history,key=lambda r:(r['foreground_canonical']['edit']+r['foreground_canonical']['f1_50'])/2)['epoch'],history=history)
        (run/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
        del model,optimizer,state
        if args.device=='cuda':
            torch.cuda.empty_cache()
    print(f'Completed: {run}',flush=True)


if __name__=='__main__':
    main()

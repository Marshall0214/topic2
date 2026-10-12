"""Explicit evaluation of a validation-selected SpecScalpel checkpoint."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from verify_specscalpel_training import (ROOT, DATA, Meter, OfficialMeter, build_model,
    ActionSegmentationDataset, ToTensor, TempDownSamp, collate_fn, PostProcessor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--checkpoint', choices=['best', 'last'], default='best')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('Simple run name required')
    run = ROOT / 'runs' / args.run
    config = json.loads((run / 'config.json').read_text())
    split_file = ROOT / 'configs/dest_compat.json'
    if hashlib.sha256(split_file.read_bytes()).hexdigest() != config['split_sha256']:
        raise RuntimeError('Split changed since training')
    split = json.loads(split_file.read_text())['splits']
    names = sorted(split['test']) if args.split == 'test' else config['selected']['val']
    assert not set(names) & set(config['selected']['train'])
    if args.split == 'test':
        assert not set(names) & set(config['selected']['val'])
    csv_root = run / f'evaluation_csv_{args.split}'
    (csv_root / 'PKU-subject').mkdir(parents=True, exist_ok=True)
    csv_name = 'test1.csv' if args.split == 'test' else 'val1.csv'
    with (csv_root / 'PKU-subject' / csv_name).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['feature', 'label', 'boundary'])
        for name in names:
            writer.writerow([f'/PKU-subject/{folder}/{name}'
                for folder in ('features', 'gt_arr', 'gt_boundary_arr')])
    dataset = ActionSegmentationDataset('PKU-subject',
        transform=Compose([ToTensor(), TempDownSamp(1)]),
        mode='test' if args.split == 'test' else 'validation',
        dataset_dir=str(DATA.parent), csv_dir=str(csv_root), augmentations=False)
    torch.set_num_threads(4)
    path = run / f'{args.checkpoint}.pt'
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['selected'] != config['selected']:
        raise RuntimeError('Checkpoint selection differs from run config')
    model = build_model().cuda().eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    joints = torch.from_numpy(np.load(ROOT / 'sources/specscalpel/assets/PKU-subject_joints_pool.npy')).cuda()
    distance = ((joints[:, None] - joints[None]) ** 2).sum(-1).sqrt()
    graph = 1 - (distance - distance.min()) / (distance.max() - distance.min())
    meters = {mode: {'foreground_canonical': Meter(), 'dest_legacy': Meter(legacy=True),
                     'protas_official': OfficialMeter()} for mode in ['raw', 'boundary_refined']}
    post = PostProcessor('refinement_with_boundary', .5)
    rows = []
    with torch.inference_mode():
        for i, sample in enumerate(DataLoader(dataset, batch_size=1, collate_fn=collate_fn)):
            logits, boundaries = model(sample['feature'].cuda(), sample['mask'].cuda(), graph)
            assert torch.isfinite(logits).all() and torch.isfinite(boundaries).all()
            preds = {'raw': logits.argmax(1).cpu().numpy()[0],
                'boundary_refined': post(logits.cpu().numpy(), boundaries=boundaries.cpu().numpy(),
                                        masks=sample['mask'].numpy())[0]}
            gt = sample['label'].numpy()[0]
            row = {'name': names[i]}
            for mode, pred in preds.items():
                for meter in meters[mode].values():
                    meter.update(pred, gt)
                one = Meter()
                one.update(pred, gt)
                row[mode] = one.scores()
            rows.append(row)
            if (i + 1) % 25 == 0:
                print(f'{args.split}: {i+1}/{len(names)}', flush=True)
    scores = {mode: {name: meter.scores() for name, meter in group.items()}
              for mode, group in meters.items()}
    result = dict(run=args.run, split=args.split, epoch=checkpoint['epoch'],
        checkpoint=args.checkpoint, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        protocol='Offline skeleton with official text priors; common local split/evaluators',
        scores=scores, per_sequence=rows)
    target = run / f'{args.split}_{args.checkpoint}_evaluation.json'
    target.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(scores, indent=2))
    print(f'Saved: {target}')


if __name__ == '__main__':
    main()

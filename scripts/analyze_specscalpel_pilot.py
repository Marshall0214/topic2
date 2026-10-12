"""Evaluate saved pilot checkpoints on selected train/val only, without augmentation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose
from verify_specscalpel_training import (ROOT, DATA, Meter, OfficialMeter, build_model,
    ActionSegmentationDataset, ToTensor, TempDownSamp, collate_fn, PostProcessor)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('Simple run name required')
    run = ROOT / 'runs' / args.run
    config = json.loads((run / 'config.json').read_text())
    torch.set_num_threads(4)
    model = build_model().cuda().eval()
    embedding = torch.from_numpy(np.load(ROOT / 'sources/specscalpel/assets/PKU-subject_joints_pool.npy')).cuda()
    distance = ((embedding[:, None] - embedding[None]) ** 2).sum(-1).sqrt()
    graph = 1 - (distance - distance.min()) / (distance.max() - distance.min())
    loaders, counts = {}, {}
    for part, mode in [('train', 'training'), ('val', 'validation')]:
        dataset = ActionSegmentationDataset('PKU-subject', mode=mode, augmentations=False,
            transform=Compose([ToTensor(), TempDownSamp(1)]), dataset_dir=str(DATA.parent),
            csv_dir=str(run / 'csv'))
        lookup = {Path(row['feature']).name: i for i, row in dataset.df.iterrows()}
        indices = [lookup[name] for name in config['selected'][part]]
        loaders[part] = DataLoader(Subset(dataset, indices), batch_size=1, collate_fn=collate_fn)
        counts[part] = np.zeros(52, dtype=np.int64)
        for name in config['selected'][part]:
            counts[part] += np.bincount(np.load(DATA / 'gt_arr' / name).astype(np.int64), minlength=52)
    unseen = np.where((counts['train'] == 0) & (counts['val'] > 0))[0].tolist()
    result = {'scope': 'Selected pilot train/val only; no test access',
        'class_coverage': {p: {'foreground_classes': (np.where(c[1:] > 0)[0] + 1).tolist(),
                               'frame_counts': c.tolist()} for p, c in counts.items()},
        'val_classes_absent_from_selected_train': unseen,
        'val_foreground_frames_in_unseen_classes_ratio':
            float(counts['val'][unseen].sum() / counts['val'][1:].sum()), 'checkpoints': {}}
    for checkpoint_name in ['best', 'last']:
        state = torch.load(run / f'{checkpoint_name}.pt', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        evaluations = {}
        for part, loader in loaders.items():
            canonical, foreground, refined_meter = Meter(), OfficialMeter(), Meter()
            predictions = np.zeros(52, dtype=np.int64)
            post = PostProcessor('refinement_with_boundary', .5)
            with torch.no_grad():
                for sample in loader:
                    logits, boundaries = model(sample['feature'].cuda(), sample['mask'].cuda(), graph)
                    pred = logits.argmax(1).cpu().numpy()[0]
                    gt = sample['label'].numpy()[0]
                    canonical.update(pred, gt)
                    foreground.update(pred, gt)
                    refined = post(logits.cpu().numpy(), boundaries=boundaries.cpu().numpy(),
                                   masks=sample['mask'].numpy())[0]
                    refined_meter.update(refined, gt)
                    predictions += np.bincount(pred, minlength=52)
            evaluations[part] = {'raw': canonical.scores(),
                'foreground_accuracy': foreground.scores()['foreground_accuracy'],
                'refined': refined_meter.scores(),
                'predicted_background_ratio': float(predictions[0] / predictions.sum()),
                'predicted_class_counts': predictions.tolist()}
        result['checkpoints'][checkpoint_name] = {'epoch': state['epoch'], 'evaluation': evaluations}
    out = ROOT / 'reports/specscalpel_audit' / f'{args.run}_analysis.json'
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

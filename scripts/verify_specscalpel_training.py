"""Real full-sequence smoke check; official preprocessing/model/full training loss.

Uses training data for optimizer updates and independent validation for metrics.
Never loads test arrays. No accuracy conclusion should be drawn from this check.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'sources/specscalpel/upstream'
sys.path[:0] = [str(UPSTREAM), str(ROOT / 'src')]
from baseline import DATA, Meter, OfficialMeter
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.transformer import ToTensor, TempDownSamp
from libs.models.SpecScalpel import Model
from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss, KLLoss
from libs.class_weight import get_pos_weight
from libs.helper import train
from libs.postprocess import PostProcessor
import libs.postprocess as upstream_postprocess


def compatible_argrelmax(prob, threshold=.7):
    """Upstream metric.argrelmax, replacing removed np.bool with np.bool_ only."""
    prob[prob < threshold] = 0.
    peak = np.concatenate([np.ones(1, dtype=np.bool_),
        (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
        np.zeros(1, dtype=np.bool_)])
    return np.where(peak)[0].tolist()


upstream_postprocess.argrelmax = compatible_argrelmax


def build_model():
    return Model(in_channel=12, n_features=64, n_classes=52, n_stages=4,
                 n_layers=10, n_refine_layers=10, n_stages_asb=2,
                 n_stages_brb=3, SFI_layer=list(range(1, 10)), dataset='PKU-subject')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--train-limit', type=int, default=4)
    parser.add_argument('--val-limit', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if Path(args.run).name != args.run or args.run in ('.', '..'):
        parser.error('Use a new simple run name')
    if min(args.train_limit, args.val_limit, args.batch_size) < 1:
        parser.error('Positive smoke-check limits required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for official training helper')
    out = ROOT / 'runs' / args.run
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    split_path = ROOT / 'configs/dest_compat.json'
    split = json.loads(split_path.read_text())['splits']
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        assert not set(split[a]) & set(split[b])
    csv_root = out / 'csv'
    (csv_root / 'PKU-subject').mkdir(parents=True)
    # No test CSV: the official dataset cannot accidentally load test here.
    for part, name in [('train', 'train1'), ('val', 'val1')]:
        with (csv_root / 'PKU-subject' / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['feature', 'label', 'boundary'])
            for seq in sorted(split[part]):
                writer.writerow([f'/PKU-subject/{folder}/{seq}'
                                 for folder in ('features', 'gt_arr', 'gt_boundary_arr')])
    datasets, chosen = {}, {}
    for part, mode, limit in [('train', 'training', args.train_limit),
                              ('val', 'validation', args.val_limit)]:
        full = ActionSegmentationDataset('PKU-subject',
            transform=Compose([ToTensor(), TempDownSamp(1)]), mode=mode,
            dataset_dir=str(DATA.parent), csv_dir=str(csv_root), augmentations=part == 'train')
        indices = list(range(len(full)))
        random.Random(args.seed + (part == 'val')).shuffle(indices)
        indices = indices[:limit]
        chosen[part] = [Path(full.df.iloc[i]['feature']).name for i in indices]
        datasets[part] = Subset(full, indices)
    assets = ROOT / 'sources/specscalpel/assets'
    manifest = json.loads((assets / 'manifest.json').read_text())
    for entry in manifest:
        assert hashlib.sha256((assets / entry['name']).read_bytes()).hexdigest() == entry['sha256']
    actions = torch.from_numpy(np.load(assets / 'PKU-subject_actions_pool.npy')).cuda()
    joints = torch.from_numpy(np.load(assets / 'PKU-subject_joints_pool.npy')).cuda()
    def graph(x):
        distance = ((x[:, None] - x[None]) ** 2).sum(-1).sqrt()
        return 1 - (distance - distance.min()) / (distance.max() - distance.min())
    action_graph, joint_graph = graph(actions), graph(joints)
    assert actions.shape == (52, 768) and joints.shape == (25, 768)
    model = build_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    pos_weight = get_pos_weight('PKU-subject', dataset_dir=str(DATA.parent),
                                csv_dir=str(csv_root), mode='training').cuda()
    cls = ActionSegmentationLoss(ce=True, focal=False, gstmse=True, gstmse_weight=1.)
    bound = BoundaryRegressionLoss(pos_weight=pos_weight)
    contrast = KLLoss().cuda()
    checks = {'component_losses': [], 'gradient_steps': []}
    def record_loss(name):
        def hook(module, inputs, value):
            assert torch.isfinite(value).all(), f'Nonfinite {name}'
            checks['component_losses'].append({'name': name, 'value': value.item()})
        return hook
    handles = [module.register_forward_hook(record_loss(name))
               for name, module in [('classification', cls), ('boundary', bound), ('contrastive', contrast)]]
    original_step = optimizer.step
    def checked_step(*a, **kw):
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads), 'Nonfinite gradient'
        norm = torch.stack([g.norm() for g in grads]).norm().item()
        assert norm > 0
        checks['gradient_steps'].append({'gradient_norm': norm, 'tensors': len(grads)})
        return original_step(*a, **kw)
    optimizer.step = checked_step
    first = datasets['train'][0]
    assert first['feature'].shape[-1] == len(first['label']) == len(first['boundary'])
    assert torch.isfinite(first['feature']).all()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    loader = DataLoader(datasets['train'], batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_fn)
    # Calls the pinned upstream helper, including CE/GSTMSE, boundary, text contrast,
    # and the adjacent-action frequency discrepancy loss without replacements.
    loss = train(loader, model, actions, action_graph, joint_graph, cls, bound,
                 contrast, torch.nn.BCEWithLogitsLoss(), .1, optimizer, 0, 'PKU-subject')
    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start
    assert np.isfinite(loss)
    for handle in handles:
        handle.remove()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, out / 'smoke.pt')
    clone = build_model().cuda()
    clone.load_state_dict(torch.load(out / 'smoke.pt', weights_only=True)['model'], strict=True)
    clone.eval()
    model.eval()
    meters = {name: Meter() for name in ('raw', 'boundary_refined')}
    supplement = OfficialMeter()
    post = PostProcessor('refinement_with_boundary', .5)
    shapes = []
    with torch.no_grad():
        for sample in DataLoader(datasets['val'], batch_size=1, collate_fn=collate_fn):
            x, mask = sample['feature'].cuda(), sample['mask'].cuda()
            logits, boundary = model(x, mask, joint_graph)
            reload_logits, _ = clone(x, mask, joint_graph)
            torch.testing.assert_close(logits, reload_logits, rtol=0, atol=0)
            assert torch.isfinite(logits).all() and torch.isfinite(boundary).all()
            raw = logits.argmax(1).cpu().numpy()[0]
            refined = post(logits.cpu().numpy(), boundaries=boundary.cpu().numpy(),
                           masks=mask.cpu().numpy())[0]
            y = sample['label'].numpy()[0]
            meters['raw'].update(raw, y)
            meters['boundary_refined'].update(refined, y)
            supplement.update(raw, y)
            shapes.append({'feature': list(x.shape), 'logits': list(logits.shape),
                           'boundary': list(boundary.shape)})
    result = dict(scope='Smoke check only; no test evaluation or accuracy conclusion',
        args=vars(args), selected=chosen, split_counts={k: len(v) for k, v in split.items()},
        split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
        environment={'torch': torch.__version__, 'torchvision': torchvision.__version__,
                     'gpu': torch.cuda.get_device_name(0)},
        official_full_training_loss=loss, train_seconds=train_seconds,
        peak_gpu_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
        boundary_pos_weight=pos_weight.item(), checks=checks, validation_shapes=shapes,
        validation={k: m.scores() for k, m in meters.items()},
        official_protas_supplement_raw=supplement.scores(), strict_checkpoint_reload=True,
        input_protocol='Official offline future displacement and relative XYZ; text prior',
        future_selection_rule='Unchanged: (raw canonical Edit + raw F1@50)/2')
    result['compatibility'] = 'Only postprocess argrelmax np.bool -> np.bool_; upstream files unchanged'
    (out / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

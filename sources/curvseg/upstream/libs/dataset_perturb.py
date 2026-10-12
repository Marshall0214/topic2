import os
from typing import Any, Dict, List, Optional

import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose

__all__ = ["ActionSegmentationDataset", "collate_fn"]

dataset_names = ["MCFS-22", "MCFS-130", "PKU-subject", "PKU-view", "LARA"]
modes = ["training", "validation", "trainval", "test"]

def get_displacements(sample):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    t = sample.shape[1]
    # Shape: C, t-1, V, M
    disps = sample[:, 1:, :, :] - sample[:, :-1, :, :]
    # Shape: C, T, V, M
    final_sample[:, start:end-1, :, :] = disps

    return final_sample

def get_relative_coordinates(sample,
                             references=(0)):
    # input: C, T, V, M
    # references=(4, 8, 12, 16)
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))
    
    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]

    C, t, V, M = sample.shape
    rel_coords = []
    #for i in range(len(references)):
    ref_loc = sample[:, :, references, :]
    coords_diff = (sample.transpose((2, 0, 1, 3)) - ref_loc).transpose((1, 2, 0, 3))
    rel_coords.append(coords_diff)
    
    # Shape: C, t, V, M 
    rel_coords = np.vstack(rel_coords)
    # Shape: C, T, V, M
    final_sample[:, start:end, :, :] = rel_coords
    return final_sample

class ActionSegmentationDataset(Dataset):
    """ Action Segmentation Dataset with boundary perturbation and detailed logging. """

    def __init__(
        self,
        dataset: str,
        transform: Optional[Compose] = None,
        mode: str = "training",
        split: int = 1,
        dataset_dir: str = "./dataset",
        csv_dir: str = "./csv",
        log_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        """
            Args:
                dataset: the name of dataset
                transform: torchvision.transforms.Compose([...])
                mode: training, validation, test
                split: which split of train, val and test do you want to use in csv files.(default:1)
                csv_dir: the path to the directory where the csv files are saved
        """

        assert (
            dataset in dataset_names
        ), "You have to choose dataset."

        if mode == "training":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            ) #Get the (NUM, 3) file, where three are features, labels, and boundary labels
        elif mode == "validation":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "val{}.csv".format(split))
            )
        elif mode == "trainval":
            df1 = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            )
            df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
            self.df = pd.concat([df1, df2])
        elif mode == "test":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "test{}.csv".format(split))
            )
        else:
            assert (
                mode in modes
            ), "You have to choose 'training', 'trainval', 'validation' or 'test' as the dataset mode."
        #    <libs.transformer.ToTensor object at 0x7f3ed3ff3550>和<libs.transformer.TempDownSamp object at 0x7f3ed402abb0>
        self.transform = transform
        self.dataset = dataset

        # Prepare perturbation info and logs
        self.log_dir = log_dir or os.path.join(csv_dir, dataset, 'perturb_logs')
        os.makedirs(self.log_dir, exist_ok=True)

        self.perturb_info: Dict[int, Dict[str, Any]] = {}
        if mode == "training":
            N = len(self.df)
            n_perturb = max(1, int(0.1 * N))
            indices = np.random.permutation(N)[:n_perturb]
            half = n_perturb // 2
            forward_idxs = indices[:half]
            backward_idxs = indices[half:]

            csv_records = []
            # Process forward perturbations
            for idx in forward_idxs:
                row = self.df.iloc[idx]
                label = np.load(row['label'])
                boundary = np.load(row['boundary'])
                # pick a boundary point
                b_pos = np.where(boundary > 0.5)[0]
                if b_pos.size == 0:
                    continue
                orig = int(np.random.choice(b_pos))
                new = min(orig + 10, boundary.shape[0] - 1)
                
                # csv_records.append([idx, row['feature'], 'forward', orig, new])
                # Log JSON with full sequences
                # feature = np.load(row['feature']).tolist()
                pert_label = label.copy()
                # forward: prev segment covers
                prev_cls = int(label[orig - 1]) if orig > 0 else int(label[orig])
                pert_label[orig:new+1] = prev_cls
                pert_boundary = boundary.copy()
                pert_boundary[orig] = 0.0
                pert_boundary[new] = 1.0
                csv_records.append([idx, row['feature'], 'backward', orig, new, \
                                    label.tolist(), pert_label.tolist(), boundary.tolist(), pert_boundary.tolist()])
                log = {
                    'sample_idx': int(idx),
                    'feature_path': row['feature'],
                    'method': 'forward',
                    'orig_boundary': int(orig),
                    'new_boundary': int(new),
                    'original_label': label.tolist(),
                    'perturbed_label': pert_label.tolist(),
                    'original_boundary': boundary.tolist(),
                    'perturbed_boundary': pert_boundary.tolist(),
                    # 'feature': feature
                }
                json.dump(log, open(os.path.join(self.log_dir, f'perturb_{idx}.json'), 'w'), indent=2)
                self.perturb_info[idx] = {"method": "forward", "orig": orig, \
                                          "new": new, 'perturbed_label': pert_label, 'perturbed_boundary': pert_boundary}
            # Process backward perturbations
            for idx in backward_idxs:
                row = self.df.iloc[idx]
                label = np.load(row['label'])
                boundary = np.load(row['boundary'])
                b_pos = np.where(boundary > 0.5)[0]
                if b_pos.size == 0:
                    continue
                orig = int(np.random.choice(b_pos))
                new = max(orig - 10, 0)
                
                
                # Log JSON
                # feature = np.load(row['feature']).tolist()
                pert_label = label.copy()
                next_cls = int(label[orig])
                pert_label[new:orig+1] = next_cls
                pert_boundary = boundary.copy()
                pert_boundary[orig] = 0.0
                pert_boundary[new] = 1.0
                csv_records.append([idx, row['feature'], 'backward', orig, new, \
                                    label.tolist(), pert_label.tolist(), boundary.tolist(), pert_boundary.tolist()])
                log = {
                    'sample_idx': int(idx),
                    'feature_path': row['feature'],
                    'method': 'backward',
                    'orig_boundary': int(orig),
                    'new_boundary': int(new),
                    'original_label': label.tolist(),
                    'perturbed_label': pert_label.tolist(),
                    'original_boundary': boundary.tolist(),
                    'perturbed_boundary': pert_boundary.tolist(),
                    # 'feature': feature
                }
                json.dump(log, open(os.path.join(self.log_dir, f'perturb_{idx}.json'), 'w'), indent=2)
                self.perturb_info[idx] = {"method": "backward", "orig": orig, \
                                          "new": new, 'perturbed_label': pert_label, 'perturbed_boundary': pert_boundary}
            # Save CSV log
            log_df = pd.DataFrame(csv_records, columns=[
                'sample_idx','feature_path','method','orig_boundary','new_boundary',\
                'original_label', 'perturbed_label', 'original_boundary', 'perturbed_boundary'
            ])
            log_df.to_csv(
                os.path.join(self.log_dir, f'boundary_perturb_log_split{split}.csv'),
                index=False
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        feature = np.load(row['feature'], allow_pickle=True).astype(np.float32)
        if (self.dataset == 'MCFS-22') or (self.dataset == 'MCFS-130'):
            feature = feature[:,:,:2] # t,v,c
            feature[:,:,0] = feature[:,:,0]/1280 - 0.5
            feature[:,:,1] = feature[:,:,1]/720 - 0.5
            feature = feature - feature[:,8:9,:]
            feature = feature.transpose(2, 1, 0) #   t,v,c--->c,v,t

        elif (self.dataset == 'PKU-subject') or (self.dataset == 'PKU-view'):
            feature = feature.reshape(-1,2,25,3).transpose(3,0,2,1) #   t,m,v,c--->c,t,v,m
            disps = get_displacements(feature)
            rel_coords = get_relative_coordinates(feature)
            feature = np.concatenate([disps, rel_coords], axis=0)
            feature = feature.transpose(3,0,2,1).reshape(12, 25, -1) #   c,t,v,m--->mc,v,t
        
        elif  (self.dataset == 'LARA'):
            disps = get_displacements(feature)
            rel_coords = get_relative_coordinates(feature)
            feature = np.concatenate([disps, rel_coords], axis=0)
            feature = feature.transpose(3,0,2,1).reshape(12, 19, -1) #   c,t,v,m--->mc,v,t
        label = np.load(row['label']).astype(np.int64)
        boundary = np.load(row['boundary']).astype(np.float32)

        # Apply perturbation if exists
        if idx in self.perturb_info:
            info = self.perturb_info[idx]
            label = info['perturbed_label']
            boundary = info['perturbed_boundary']

        if self.transform:
            feature, label, boundary = self.transform([feature, label, boundary])

        return {
            'feature': feature,
            'label': label,
            'boundary': boundary,
            'feature_path': row['feature']
        }


def collate_fn(sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_length = max(s['feature'].shape[2] for s in sample)
    feat_list, label_list, path_list, boundary_list, length_list = [], [], [], [], []
    for s in sample:
        feat, lbl, bnd = s['feature'], s['label'], s['boundary']
        T = feat.shape[2]
        pad = max_length - T
        length_list.append(T)
        if pad>0:
            feat = F.pad(feat, (0,pad), 'constant', 0)
            lbl  = F.pad(lbl,  (0,pad), 'constant', 255)
            bnd  = F.pad(bnd,  (0,pad), 'constant', 0)
        bnd = bnd.unsqueeze(0)
        feat_list.append(feat)
        label_list.append(lbl)
        boundary_list.append(bnd)
        path_list.append(s['feature_path'])
    features   = torch.stack(feat_list,   0)
    labels     = torch.stack(label_list,  0)
    boundaries = torch.stack(boundary_list,0)
    masks      = torch.tensor([[[1 if i< L else 0 for i in range(max_length)]] for L in length_list], dtype=torch.bool)
    return {
        'feature': features,
        'label':   labels,
        'boundary':boundaries,
        'feature_path': path_list,
        'mask':    masks
    }

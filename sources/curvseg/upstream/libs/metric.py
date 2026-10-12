# https://github.com/cthincsl/TemporalConvthutionalNetworks/blob/master/code/metrics.py
# Score metric for action segmentation was originally written by cthincs1

import copy
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from torchvision.ops import box_iou


def get_segments(
    frame_wise_label: np.ndarray,
    id2class_map: Dict[int, str],
    bg_class: str = "background",
) -> Tuple[List[int], List[int], List[int]]:
    """
    Args:
        frame-wise label: frame-wise prediction or ground truth. 1D numpy array
    Return:
        segment-label array: list (excluding background class)
        start index list
        end index list
    """

    labels = []
    starts = []
    ends = []

    frame_wise_label = [
        id2class_map[frame_wise_label[i]] for i in range(len(frame_wise_label))
    ]

    # get class, start index and end index of segments
    # background class is excluded
    last_label = frame_wise_label[0]
    if frame_wise_label[0] != bg_class:
        labels.append(frame_wise_label[0])
        starts.append(0)

    for i in range(len(frame_wise_label)):
        # if action labels change
        if frame_wise_label[i] != last_label:
            # if label change from one class to another class
            # it's an action starting point
            if frame_wise_label[i] != bg_class:
                labels.append(frame_wise_label[i])
                starts.append(i)

            # if label change from background to a class
            # it's not an action end point.
            if last_label != bg_class:
                ends.append(i)

            # update last label
            last_label = frame_wise_label[i]

    if last_label != bg_class:
        ends.append(i)

    return labels, starts, ends


def levenshtein(pred: List[int], gt: List[int], norm: bool = True) -> float:
    """
    Levenshtein distance(Edit Distance)
    Args:
        pred: segments list
        gt: segments list
    Return:
        if norm == True:
            (1 - average_edit_distance) * 100
        else:
            edit distance
    """

    n, m = len(pred), len(gt)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # insertion
                dp[i][j - 1] + 1,  # deletion
                dp[i - 1][j - 1] + cost,
            )  # replacement

    if norm:
        score = (1 - dp[n][m] / max(n, m)) * 100
    else:
        score = dp[n][m]

    return score


def get_n_samples(
    p_label: List[int],
    p_start: List[int],
    p_end: List[int],
    g_label: List[int],
    g_start: List[int],
    g_end: List[int],
    iou_threshold: float,
    bg_class: List[str] = ["background"],
) -> Tuple[int, int, int]:
    """
    Args:
        p_label, p_start, p_end: return values of get_segments(pred)
        g_label, g_start, g_end: return values of get_segments(gt)
        threshold: threshold (0.1, 0.25, 0.5)
        bg_class: background class
    Return:
        tp: true positive
        fp: false positve
        fn: false negative
    """

    tp = 0
    fp = 0
    hits = np.zeros(len(g_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], g_end) - np.maximum(p_start[j], g_start)
        union = np.maximum(p_end[j], g_end) - np.minimum(p_start[j], g_start)
        IoU = (1.0 * intersection / union) * (
            [p_label[j] == g_label[x] for x in range(len(g_label))]
        )
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= iou_threshold and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = len(g_label) - sum(hits)

    return float(tp), float(fp), float(fn)


import torch

def calculate_iou_for_sequence(
    p_label: List[int],
    p_start: List[int],
    p_end: List[int],
    g_label: List[int],
    g_start: List[int],
    g_end: List[int],
) -> List[float]:
    """
    Calculate IoU values for each predicted and ground truth sequence segment, 
    considering only matching labels.
    
    Args:
        p_label, p_start, p_end: Predicted sequence labels, start times, and end times.
        g_label, g_start, g_end: Ground truth sequence labels, start times, and end times.
    
    Return:
        class iou: list of IoU values for each class
    """
    
    class2pred = {}
    for i in range(len(p_label)):
        class2pred[p_label[i]] = class2pred.get(p_label[i], []) + [[p_start[i], 0, p_end[i], 1]]
    for key in class2pred:
        class2pred[key] = torch.tensor(class2pred[key])
    class2gt = {}
    for i in range(len(g_label)):
        class2gt[g_label[i]] = class2gt.get(g_label[i], []) + [[g_start[i], 0, g_end[i], 1]]
    for key in class2gt:
        class2gt[key] = torch.tensor(class2gt[key])
    class_ious = {}
    class_length = {}
    for key in class2gt:
        class_length[key] = class_length.get(key, []) + [len(it) for it in class2gt[key]]
        if key not in class2pred:
            class_ious[key] = class_ious.get(key, []) + [0 for _ in range(len(class2gt[key]))]

        else:
            iou = box_iou(class2pred[key], class2gt[key]).numpy()
            iou_x = iou.max(axis=0)
            assert iou_x.shape[0] == len(class2gt[key]), "iou_x.shape[0]!= len(class2gt[key])"
            # iou_x to list
            iou_x = iou_x.tolist()
            class_ious[key] = class_ious.get(key, []) + iou_x
    return class_ious, class_length

def calculate_biou_for_sequence(
    p_start: List[int],
    p_end: List[int],
    g_label: List[int],
    g_start: List[int],
    g_end: List[int],
) -> List[float]:
    """
    Calculate IoU values for each predicted and ground truth sequence segment, 
    considering only matching labels.
    
    Args:
        p_label, p_start, p_end: Predicted sequence labels, start times, and end times.
        g_label, g_start, g_end: Ground truth sequence labels, start times, and end times.
    
    Return:
        class iou: list of IoU values for each class
    """
    
    pred = []
    for i in range(len(p_start)):
        pred.append([p_start[i], 0, p_end[i], 1])
    
    
    pred = torch.tensor(pred)
    

    class2gt = {}
    for i in range(len(g_label)):
        class2gt[g_label[i]] = class2gt.get(g_label[i], []) + [[g_start[i], 0, g_end[i], 1]]
    for key in class2gt:
        class2gt[key] = torch.tensor(class2gt[key])
    class_ious = {}
    # class_length = {}
    for key in class2gt:
        # class_length[key] = class_length.get(key, []) + [len(it) for it in class2gt[key]]
        if pred.shape[0] == 0:
            class_ious[key] = class_ious.get(key, []) + [0 for _ in range(len(class2gt[key]))]
            continue
        iou = box_iou(pred, class2gt[key]).numpy()
        iou_x = iou.max(axis=0)
        assert iou_x.shape[0] == len(class2gt[key]), "iou_x.shape[0]!= len(class2gt[key])"
        # iou_x to list
        iou_x = iou_x.tolist()
        class_ious[key] = class_ious.get(key, []) + iou_x
    # return class_ious, class_length
    return class_ious

class ScoreMeter(object):
    def __init__(
        self,
        id2class_map: Dict[int, str],
        iou_thresholds: Tuple[float] = (0.1, 0.25, 0.5),
        ignore_index: int = 255,
    ) -> None:

        self.iou_thresholds = iou_thresholds  # threshold for f score(0.1, 0.25, 0.5, 0.75, 0.9)
        self.ignore_index = ignore_index #255
        self.id2class_map = id2class_map #map
        self.edit_score = 0
        self.tp = [0 for _ in range(len(iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.n_classes = len(self.id2class_map)
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        mask = (gt >= 0) & (gt < self.n_classes)
        hist = np.bincount(
            self.n_classes * gt[mask].astype(int) + pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(
        self,
        outputs: np.ndarray,
        gts: np.ndarray,
        boundaries: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            outputs: np.array. shape(N, C, T)
                the model output for boundary prediciton
            gt: np.array. shape(N, T)
                Ground Truth for boundary
        """
        if len(outputs.shape) == 3:
            preds = outputs.argmax(axis=1)
        elif len(outputs.shape) == 2:
            preds = copy.copy(outputs)

        for pred, gt in zip(preds, gts):
            pred = pred[gt != self.ignore_index]
            gt = gt[gt != self.ignore_index]

            for lt, lp in zip(pred, gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

            self.n_videos += 1
            # count the correct frame
            self.n_frames += len(pred)
            for i in range(len(pred)):
                if pred[i] == gt[i]:
                    self.n_correct += 1

            # calculate the edit distance
            p_label, p_start, p_end = get_segments(pred, self.id2class_map)
            g_label, g_start, g_end = get_segments(gt, self.id2class_map)

            self.edit_score += levenshtein(p_label, g_label, norm=True)

            for i, th in enumerate(self.iou_thresholds):
                tp, fp, fn = get_n_samples(
                    p_label, p_start, p_end, g_label, g_start, g_end, th
                )
                self.tp[i] += tp
                self.fp[i] += fp
                self.fn[i] += fn

    def get_scores(self) -> Tuple[float, float, float]:
        """
        Return:
            Accuracy
            Normlized Edit Distance
            F1 Score of Each Threshold
        """
        # print("self.n_correct", self.n_correct)
        # print("self.n_frames", self.n_frames)
        # accuracy
        acc = 100 * float(self.n_correct) / self.n_frames #acc

        # edit distance
        edit_score = float(self.edit_score) / self.n_videos #edit

        # F1 Score
        f1s = []
        for i in range(len(self.iou_thresholds)): #F1@{10,25,50,75,90}
            precision = self.tp[i] / float(self.tp[i] + self.fp[i])
            recall = self.tp[i] / float(self.tp[i] + self.fn[i])

            f1 = 2.0 * (precision * recall) / (precision + recall + 1e-7)
            f1 = np.nan_to_num(f1) * 100

            f1s.append(f1)

        # Accuracy, Edit Distance, F1 Score
        return acc, edit_score, f1s

    def return_confusion_matrix(self) -> np.ndarray:
        return self.confusion_matrix

    def save_scores(self, save_path: str) -> None:
        acc, edit_score, segment_f1s = self.get_scores()

        # save log
        columns = ["cls_acc", "edit"]
        data_dict = {
            "cls_acc": [acc],
            "edit": [edit_score],
        }

        for i in range(len(self.iou_thresholds)):
            key = "segment f1s@{}".format(self.iou_thresholds[i])
            columns.append(key)
            data_dict[key] = [segment_f1s[i]]

        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)

    def save_confusion_matrix(self, save_path: str) -> None:
        with open(save_path, "w") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerows(self.confusion_matrix)

    def reset(self) -> None:
        self.edit_score = 0
        self.tp = [0 for _ in range(len(self.iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(self.iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(self.iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def argrelmax(prob: np.ndarray, threshold: float = 0.7) -> List[int]:
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold

    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0

    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )

    peak_idx = np.where(peak)[0].tolist()

    return peak_idx


class BoundaryScoreMeter(object):
    def __init__(self, tolerance=5, boundary_threshold=0.7):
        # max distance of the frame which can be regarded as correct
        self.tolerance = tolerance #5

        # threshold of the boundary value which can be regarded as action boundary
        self.boundary_threshold = boundary_threshold #0.5
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0

    def update(self, preds, gts, masks):
        """
        Args:
            preds: np.array. the model output(N, T)
            gts: np.array. boudnary ground truth array (N, T)
            masks: np.array. np.bool. valid length for each video (N, T)
        Return:
            Accuracy
            Boundary F1 Score
        """

        for pred, gt, mask in zip(preds, gts, masks):
            # ignore invalid frames
            # print(pred.shape, gt.shape, mask.shape)
            pred = pred[mask]
            gt = gt[mask]

            pred_idx = argrelmax(pred, threshold=self.boundary_threshold)
            gt_idx = argrelmax(gt, threshold=self.boundary_threshold)

            n_frames = pred.shape[0]
            tp = 0.0
            fp = 0.0
            fn = 0.0

            hits = np.zeros(len(gt_idx))

            # calculate true positive, false negative, false postive, true negative
            for i in range(len(pred_idx)):
                dist = np.abs(np.array(gt_idx) - pred_idx[i])
                min_dist = np.min(dist)
                idx = np.argmin(dist)

                if min_dist <= self.tolerance and hits[idx] == 0:
                    tp += 1 #tp
                    hits[idx] = 1
                else:
                    fp += 1

            fn = len(gt_idx) - sum(hits)
            tn = n_frames - tp - fp - fn

            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.n_frames += n_frames
            self.n_correct += tp + tn

    def get_scores(self):
        """
        Return:
            Accuracy
            Boundary F1 Score
        """

        # accuracy
        acc = 100 * self.n_correct / self.n_frames #acc

        # Boudnary F1 Score
        precision = self.tp / float(self.tp + self.fp) # gassian
        recall = self.tp / float(self.tp + self.fn)

        f1s = 2.0 * (precision * recall) / (precision + recall + 1e-7)
        f1s = np.nan_to_num(f1s) * 100

        # Accuracy, Edit Distance, F1 Score
        return acc, precision * 100, recall * 100, f1s

    def save_scores(self, save_path: str) -> None:
        acc, precision, recall, f1s = self.get_scores()

        # save log
        columns = ["bound_acc", "precision", "recall", "bound_f1s"]
        data_dict = {
            "bound_acc": [acc],
            "precision": [precision],
            "recall": [recall],
            "bound_f1s": [f1s],
        }

        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)

    def reset(self):
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
    

import matplotlib.pyplot as plt
import os

class IOUMeter(object):

    def __init__(self, id2class_map: Dict[int, str], 
                 max_len: int = 9000,
                 thresholds: Tuple[float] = (0.12, 0.25, 0.5, 0.75),
                 ignore_index: int = 255) -> None:
        self.id2class_map = id2class_map
        self.ignore_index = ignore_index
        self.n_classes = len(self.id2class_map)
        self.thresholds = thresholds
        self.class_iou = {id2class_map[i]:[] for i in range(self.n_classes)}
        self.seqlen_iou = [[] for _ in range(max_len)]
        self.seglen_iou = [[] for _ in range(max_len)]
        self.class_biou = [{id2class_map[i]:[] for i in range(self.n_classes)} \
                           for _ in range(len(self.thresholds))]
        self.class_ciou = [{id2class_map[i]:[] for i in range(self.n_classes)}\
                           for _ in range(len(self.thresholds))]
        # self.seqlen = []
        self.seglen = {id2class_map[i]:[] for i in range(self.n_classes)}
        self.seq_iou = []
        self.seq_pair = []

    def update(self, outputs: np.ndarray, gts: np.ndarray, 
               outputs_boundary: np.ndarray, outputs_curves: np.ndarray) -> None:
        """
        Args:
            outputs: np.array. shape(N, C, T)
                the model output for boundary prediciton
            gt: np.array. shape(N, T)
                Ground Truth for boundary
            outputs_boundary: np.array. shape(N, T)
            outputs_curves: np.array. shape(N, C, T)
        """
        if len(outputs.shape) == 3:
            preds = outputs.argmax(axis=1)
        elif len(outputs.shape) == 2:
            preds = copy.copy(outputs)
        outputs_boundary = np.squeeze(outputs_boundary, axis=1)

        for pred, gt, pb, pc in zip(preds, gts, outputs_boundary, outputs_curves):
            pred = pred[gt != self.ignore_index]
            gt = gt[gt != self.ignore_index]
            pb = pb[gt != self.ignore_index]
            pc = pc[gt != self.ignore_index]

            
            
            p_label, p_start, p_end = get_segments(pred, self.id2class_map)
            g_label, g_start, g_end = get_segments(gt, self.id2class_map)
            cious, clength = calculate_iou_for_sequence(p_label, p_start, p_end,\
                                                         g_label, g_start, g_end)
            
            b_starts = []
            b_ends = []
            c_starts = []
            c_ends = []
            for id, th in enumerate(self.thresholds):
                pb_th_indices = np.where(pb >= th)[0]
                pc_th_indices = np.where(pc >= th)[0]

                b_start = pb_th_indices[:-1]
                b_end = pb_th_indices[1:]
                c_start = pc_th_indices[:-1]
                c_end = pc_th_indices[1:]
                b_starts.append(b_start)
                b_ends.append(b_end)
                c_starts.append(c_start)
                c_ends.append(c_end)
                
                biou = calculate_biou_for_sequence(b_start, b_end, \
                                            g_label, g_start, g_end)
                
                curve_iou = calculate_biou_for_sequence(c_start, c_end, \
                                                    g_label, g_start, g_end)
                
                for ky in biou:
                    self.class_biou[id][ky] += biou[ky]
                for ky in curve_iou:
                    self.class_ciou[id][ky] += curve_iou[ky]
            
            # calculate the iou for each class
            for id, i in enumerate(cious.keys()):
                self.class_iou[i] += cious[i]
             
                self.seglen[i] += clength[i]
                for j in range(len(cious[i])):
                    self.seglen_iou[clength[i][j]] += [cious[i][j]]
            
            ious = []
            for key in cious:
                ious += cious[key]
            # calculate the mean iou
            mean_iou = np.mean(ious)
            self.seqlen_iou[len(pred)] += [mean_iou]
            self.seq_iou += [mean_iou]
            self.seq_pair += [{
                "p_label": p_label,
                "p_start": p_start,
                "p_end": p_end,
                "g_label": g_label,
                "g_start": g_start,
                "g_end": g_end,
                "pred": pred,
                "gt": gt,
                "b_starts": b_starts,
                "b_ends": b_ends,
                "c_starts": c_starts,
                "c_ends": c_ends,
            }]
            
            # for i, th in enumerate(self.iou_thresholds):
            #     tp, fp, fn = get_n_samples(
            #         p_label, p_start, p_end, g_label, g_start, g_end, th
            #     )
    def get_scores(self) -> Tuple[float, float, float]:
        avg_class_iou = {}
        for i in range(self.n_classes):
            if len(self.class_iou[self.id2class_map[i]]) > 0:
                avg_class_iou[self.id2class_map[i]] = np.mean(self.class_iou[self.id2class_map[i]])

        all_ious = [iou for sublist in self.class_iou for iou in self.class_iou[sublist]]
        # print(all_ious)
        mean_iou = np.mean(all_ious) if len(all_ious) > 0 else 0

        avg_class_biou = {}
        for id, th in enumerate(self.thresholds):
            avg_class_biou[th] = {}
            for i in range(self.n_classes):
                if len(self.class_biou[id][self.id2class_map[i]]) > 0:
                    avg_class_biou[th][self.id2class_map[i]] = \
                        np.mean(self.class_biou[id][self.id2class_map[i]])
        avg_class_ciou = {}
        for id, th in enumerate(self.thresholds):
            avg_class_ciou[th] = {}
            for i in range(self.n_classes):
                if len(self.class_ciou[id][self.id2class_map[i]]) > 0:
                    avg_class_ciou[th][self.id2class_map[i]] = \
                        np.mean(self.class_ciou[id][self.id2class_map[i]])
        mean_biou = []
        mean_ciou = []
        for id, th in enumerate(self.thresholds):
            all_bious = []
            all_cious = []
            for ky in self.class_biou[id]:
                if ky == "NONE":
                    continue
                all_bious += [iou for iou in self.class_biou[id][ky]]
                all_cious += [iou for iou in self.class_ciou[id][ky]]
        
            mean_biou.append(np.mean(all_bious) if len(all_bious) > 0 else 0)
            mean_ciou.append(np.mean(all_cious) if len(all_cious) > 0 else 0)
        return mean_iou, avg_class_iou, mean_biou, avg_class_biou, mean_ciou, avg_class_ciou

    def plot_iou(self, save_path: str) -> None:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        mean_iou, avg_class_iou, mean_biou, avg_class_biou, mean_ciou, avg_class_ciou = self.get_scores()

        sorted_classes = sorted(avg_class_iou.items(), key=lambda x: x[1])
        classes, iou_values = zip(*sorted_classes)

        plt.figure(figsize=(16, 6))
        bars = plt.bar(classes, iou_values, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Average IoU')
        plt.title('Average IoU per Class')
        plt.xticks(rotation=90)

        # 添加平均 IoU 线
        plt.axhline(mean_iou, color='purple', linestyle='--', label=f'Mean IoU: {mean_iou:.2f}')
        plt.legend()

        # 在每个条形上方添加 IoU 值标签
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')

        # 保存图像到指定路径
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        for id, th in enumerate(self.thresholds):
            save_path_th = os.path.join(os.path.dirname(save_path), f"biou_th_{th:.2f}.png")
            sorted_classes_biou = sorted(avg_class_biou[th].items(), key=lambda x: x[1])
            classes, biou_values = zip(*sorted_classes_biou)
            plt.figure(figsize=(16, 6))
            bars = plt.bar(classes, biou_values, color='orange')
            plt.xlabel('Class')
            plt.ylabel('Average Biou')
            plt.title(f'Average Biou per Class (Boundary threshold: {th:.2f})')
            plt.xticks(rotation=90)
            # 添加平均 Biou 线
            plt.axhline(mean_biou[id], color='purple', linestyle='--', label=f'Mean Biou: {mean_biou[id]:.2f}')
            plt.legend()
            # 在每个条形上方添加 Biou 值标签
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
                # 保存图像到指定路径
            plt.savefig(save_path_th, bbox_inches='tight')
            plt.close()

            save_path_th = os.path.join(os.path.dirname(save_path), f"ciou_th_{th:.2f}.png")
            sorted_classes_ciou = sorted(avg_class_ciou[th].items(), key=lambda x: x[1])
            classes, ciou_values = zip(*sorted_classes_ciou)
            plt.figure(figsize=(16, 6))
            bars = plt.bar(classes, ciou_values, color='green')
            plt.xlabel('Class')
            plt.ylabel('Average Ciou')
            plt.title(f'Average Ciou per Class (Boundary threshold: {th:.2f})')
            plt.xticks(rotation=90)
            # 添加平均 Ciou 线
            plt.axhline(mean_ciou[id], color='purple', linestyle='--', label=f'Mean Ciou: {mean_ciou[id]:.2f}')
            plt.legend()
            # 在每个条形上方添加 Ciou 值标签
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom')
                # 保存图像到指定路径
            plt.savefig(save_path_th, bbox_inches='tight')
            plt.close()


        # 绘制iou最低的前k个clip
        
        k = 234
        sorted_seq_iou = sorted(self.seq_iou)
        top_k_seq_iou = sorted_seq_iou[:k]
        top_k_seq_pair = [self.seq_pair[i] for i in np.argsort(self.seq_iou)[:k]]
        # print(top_k_seq_pair)
        # exit()
        save_path = os.path.dirname(save_path)
        save_path = os.path.join(save_path, f"top_{k}_iou")
        # mkdir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for id in range(len(top_k_seq_iou)):
            fig, ax = plt.subplots(1, figsize=(70,6))
            save_path_i = os.path.join(save_path, f"{id:03d}_iou_{sorted_seq_iou[id]:.2f}.png")
            for sid, g_label in enumerate(top_k_seq_pair[id]["g_label"]):
                if g_label == "NONE":
                    continue
                i, j = top_k_seq_pair[id]["g_start"][sid], top_k_seq_pair[id]["g_end"][sid]
                ax.barh(g_label, j-i, left=i, height=0.5, color='blue', alpha=0.5, label='Ground Truth' if sid == 0 else "")
            
            for sid, p_label in enumerate(top_k_seq_pair[id]["p_label"]):
                if p_label == "NONE":
                    continue
                i, j = top_k_seq_pair[id]["p_start"][sid], top_k_seq_pair[id]["p_end"][sid]
                ax.barh(p_label, j-i, left=i, height=0.5, color='yellow', alpha=0.5, label='Prediction' if sid == 0 else "")
            
            # turn np.array to list
            # b_ends
            # p_boundaries = list(set(top_k_seq_pair[id]["b_starts"])\
            #                     +set(top_k_seq_pair[id]["b_ends"]))
            # p_curves = list(set(top_k_seq_pair[id]["c_starts"])\
            #                     +set(top_k_seq_pair[id]["c_ends"]))
            # x_min, x_max = ax.get_xlim()
            b_starts = top_k_seq_pair[id]["b_starts"][0].tolist()
            b_ends = top_k_seq_pair[id]["b_ends"][0].tolist()
            c_starts = top_k_seq_pair[id]["c_starts"][0].tolist()
            c_ends = top_k_seq_pair[id]["c_ends"][0].tolist()
            g_starts = top_k_seq_pair[id]["g_start"]
            g_ends = top_k_seq_pair[id]["g_end"]
            
            for sid, (l, r) in enumerate(zip(b_starts, b_ends)):
                ax.barh("bound_pred_0.5", 1, left=l, height=0.5, color='purple', alpha=0.5, label='bound_pred' if sid == 0 else "")
                if sid == len(b_starts) - 1:
                    ax.barh("bound_pred_0.5", 1, left=r, height=0.5, color='purple', alpha=0.5)
            for sid, (l, r) in enumerate(zip(c_starts, c_ends)):
                ax.barh("curve_pred_0.5", 1, left=l, height=0.5, color='orange', alpha=0.5, label='curve_pred' if sid == 0 else "")
                if sid == len(c_starts) - 1:
                    ax.barh("curve_pred_0.5", 1, left=r, height=0.5, color='orange', alpha=0.5)
            g_starts = np.array(g_starts)
            g_ends = np.array(g_ends)
            ax.barh("gt", 1, left=g_starts, height=0.5, color="red")
            ax.set_yticks(list(top_k_seq_pair[id]["g_label"]) + list(top_k_seq_pair[id]["p_label"] + ["bound_pred_0.5", "curve_pred_0.5", "gt"]))
            ax.set_yticklabels(list(top_k_seq_pair[id]["g_label"]) + list(top_k_seq_pair[id]["p_label"] + ["bound_pred_0.5", "curve_pred_0.5", "gt"]))
            ax.set_xlabel("Time (frames)")
            ax.set_title(f"IoU: {sorted_seq_iou[id]:.2f}")
            ax.legend(loc='upper right')
            plt.savefig(save_path_i, bbox_inches='tight')
            plt.close()
            

            


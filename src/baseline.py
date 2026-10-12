"""Shared PKU input, metrics and explicitly attributed local baseline adapters."""
import importlib.util
import json
from pathlib import Path
import pickle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r'D:\AQA\online_skeleton_action_segmentation\official_data\dataset\PKU-subject')
spec = importlib.util.spec_from_file_location('pinned_protas', ROOT/'sources/protas/upstream/model.py')
PROTAS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PROTAS)


def skeleton_features(raw):
    """(T,150) -> (300,T): per-person relative XYZ then backward XYZ."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 150 or not np.isfinite(raw).all():
        raise ValueError('Expected finite (T,150) skeleton array')
    p = raw.reshape(-1,2,25,3)
    valid = np.any(p != 0, axis=(2,3))
    relative = (p - p[:,:,:1,:]) * valid[:,:,None,None]
    velocity = np.zeros_like(p)
    velocity[1:] = (p[1:] - p[:-1]) * (valid[1:] & valid[:-1])[:,:,None,None]
    return np.concatenate([relative.reshape(len(p),150),velocity.reshape(len(p),150)],axis=1).T.copy()


def load_sequence(name, data=DATA):
    x = skeleton_features(np.load(data/'features'/name, allow_pickle=False))
    y = np.load(data/'gt_arr'/name, allow_pickle=False)
    if y.ndim != 1 or len(y) != x.shape[1] or not np.isfinite(y).all() or not (y == np.floor(y)).all() or not ((0 <= y) & (y < 52)).all():
        raise ValueError(f'Invalid label array: {name}')
    return x, y.astype(np.int64)


def progress_targets(y, classes=52):
    progress = np.zeros((classes,len(y)),dtype=np.float32)
    labels, starts, ends = segments(y, exclude_background=False)
    for label,start,end in zip(labels,starts,ends):
        if label != 0:
            progress[label,start:end] = np.arange(1,end-start+1,dtype=np.float32)/(end-start)
    return progress


def make_graph(labels, output, classes=52):
    """Exactly selected training sequences, never scan a global label directory."""
    pre = np.zeros((classes,classes),dtype=np.float32)
    suc = np.zeros_like(pre)
    count = np.zeros(classes,dtype=np.float32)
    for y in labels:
        # Match ProTAS transcript convention: remove background then merge repeats.
        foreground = y[y != 0]
        transcript,_,_ = segments(foreground,exclude_background=False)
        for a in transcript:
            count[a] += 1
        for a,b in zip(transcript[:-1],transcript[1:]):
            pre[a,b] += 1
            suc[b,a] += 1
    graph = {'matrix_pre':pre/np.maximum(count[None,:],1e-5),
             'matrix_suc':suc/np.maximum(count[None,:],1e-5)}
    with Path(output).open('wb') as f:
        pickle.dump(graph,f)
    return graph


def segments(y, exclude_background=True, legacy=False):
    y = np.asarray(y)
    if not len(y):
        return [],[],[]
    starts = np.r_[0,np.flatnonzero(y[1:] != y[:-1])+1]
    ends = np.r_[starts[1:],len(y)]
    if legacy:
        ends[-1] = len(y)-1
    keep = y[starts] != 0 if exclude_background else np.ones(len(starts),dtype=bool)
    return y[starts][keep].tolist(),starts[keep].tolist(),ends[keep].tolist()


def edit_score(a,b):
    if not a and not b:
        return 100.
    previous = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row = [i]
        for j,z in enumerate(b,1):
            row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=z)))
        previous = row
    return 100.*(1.-previous[-1]/max(len(a),len(b)))


class Meter:
    def __init__(self, legacy=False):
        self.legacy = legacy
        self.tp = np.zeros(3)
        self.fp = np.zeros(3)
        self.fn = np.zeros(3)
        self.correct = self.frames = self.videos = 0
        self.edit = 0.

    def update(self,pred,y):
        pred,y = np.asarray(pred),np.asarray(y)
        if pred.ndim != 1 or pred.shape != y.shape or not len(y):
            raise ValueError('Nonempty aligned 1D labels required')
        self.correct += int((pred == y).sum())
        self.frames += len(y)
        self.videos += 1
        pl,ps,pe = segments(pred,not self.legacy,self.legacy)
        gl,gs,ge = segments(y,not self.legacy,self.legacy)
        self.edit += edit_score(pl,gl)
        for k,threshold in enumerate([.1,.25,.5]):
            hits = np.zeros(len(gl),dtype=bool)
            for label,start,end in zip(pl,ps,pe):
                if not gl:
                    self.fp[k] += 1
                    continue
                intersection = np.minimum(end,ge)-np.maximum(start,gs)
                union = np.maximum(end,ge)-np.minimum(start,gs)
                iou = np.divide(intersection,union,out=np.zeros(len(gl),dtype=float),where=np.asarray(union)>0)* (np.asarray(gl)==label)
                best = int(iou.argmax())
                if iou[best]>=threshold and not hits[best]:
                    self.tp[k] += 1
                    hits[best] = True
                else:
                    self.fp[k] += 1
            self.fn[k] += int((~hits).sum())

    def scores(self):
        f1 = 100.*np.divide(2*self.tp,2*self.tp+self.fp+self.fn,out=np.zeros(3),where=(2*self.tp+self.fp+self.fn)>0)
        acc = 100.*self.correct/max(self.frames,1)
        edit = self.edit/max(self.videos,1)
        return dict(accuracy=acc,edit=edit,
                    f1_10=float(f1[0]),f1_25=float(f1[1]),f1_50=float(f1[2]),
                    avg=float((acc+edit+f1.sum())/5),sequences=self.videos,frames=self.frames)


class CausalStage(nn.Module):
    """MS-TCN structure using the pinned ProTAS causal residual layer."""
    def __init__(self,layers,width,dim,classes):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(dim,width,1)
        self.layers = nn.ModuleList([PROTAS.DilatedResidualLayer(2**i,width,width,causal=True) for i in range(layers)])
        self.conv_out = nn.Conv1d(width,classes,1)

    def forward(self,x,mask):
        x = self.conv_1x1(x)
        for layer in self.layers:
            x = layer(x,mask)
        return self.conv_out(x)*mask


class CausalMSTCN(nn.Module):
    def __init__(self,stages=4,layers=10,width=64,dim=300,classes=52):
        super().__init__()
        self.stage1 = CausalStage(layers,width,dim,classes)
        self.stages = nn.ModuleList([CausalStage(layers,width,classes,classes) for _ in range(stages-1)])

    def forward(self,x,mask):
        out = self.stage1(x,mask)
        outputs = [out]
        for stage in self.stages:
            out = stage(out.softmax(1)*mask,mask)
            outputs.append(out)
        return torch.stack(outputs), None


def build_model(kind,graph_path=None,stages=4,layers=10,width=64,graph_reference='independent_copy',task_graph='learnable'):
    if kind == 'mstcn':
        return CausalMSTCN(stages,layers,width)
    if task_graph not in ['learnable','fixed']:
        raise ValueError('Unknown task graph mode')
    model = PROTAS.MultiStageModel(stages,layers,width,300,52,causal=True,use_graph=True,
                                  init_graph_path=str(graph_path),learnable=task_graph=='learnable')
    if task_graph == 'fixed':
        return model
    if graph_reference == 'official':
        return model
    if graph_reference != 'independent_copy':
        raise ValueError('Unknown ProTAS graph reference mode')
    graph = model.graph_learner
    # Official original Parameters alias trainable storage: make fixed snapshots
    # true buffers so graph regularization has the intended nonzero reference.
    for name in ['matrix_pre_original','matrix_suc_original']:
        value = getattr(graph,name).detach().clone()
        delattr(graph,name)
        graph.register_buffer(name,value)
    return model


class OfficialMeter(Meter):
    """Official ProTAS Edit plus foreground accuracy; ID0 maps to background."""
    def __init__(self):
        super().__init__()
        self.foreground_correct = self.foreground_frames = 0

    def update(self,pred,y):
        pred,y = np.asarray(pred),np.asarray(y)
        old_edit = self.edit
        super().update(pred,y)
        # Official eval.py removes background BEFORE merging repeated labels.
        pl,_,_ = segments(pred[pred != 0],exclude_background=False)
        gl,_,_ = segments(y[y != 0],exclude_background=False)
        self.edit = old_edit + edit_score(pl,gl)
        foreground = y != 0
        self.foreground_frames += int(foreground.sum())
        self.foreground_correct += int(((pred == y) & foreground).sum())

    def scores(self):
        result = super().scores()
        result['foreground_accuracy'] = (100*self.foreground_correct/self.foreground_frames
                                         if self.foreground_frames else None)
        result['foreground_frames'] = self.foreground_frames
        return result


def predict_logits(model,kind,x,mask):
    outputs,progress = model(x,mask)
    logits = outputs[-1]
    if kind == 'protas':
        logits = model.graph_learner.inference(logits,progress[-1])
    return logits


def training_loss(model,kind,x,y,progress,mask):
    outputs,progress_outputs = model(x,mask)
    loss = outputs.new_zeros(())
    for out in outputs:
        loss = loss + F.cross_entropy(out,y)
        if out.shape[-1]>1:
            difference = (F.log_softmax(out[:,:,1:],dim=1)-F.log_softmax(out.detach()[:,:,:-1],dim=1)).square().clamp(0,16)
            loss = loss + .15*difference.mean()
    if kind == 'protas':
        loss = loss + sum(F.mse_loss(p,progress) for p in progress_outputs)
        loss = loss + .1*model.graph_learner(outputs[-1],progress_outputs[-1])
    return loss

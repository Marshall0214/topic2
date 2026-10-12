import torch
import random
import numpy as np
from torch.utils.data import Dataset

class SegmentSwapTransform:
    """
    根据 groundtruth 分类标签对序列做分段随机交换。

    输入:
        features: Tensor 或 ndarray, shape [..., T] 或 [T, ...]
        cls_labels: 1D Tensor 或 ndarray, 长度 T，标记每帧的类别
        boundary_labels: 可选，不需要时可为 None
    输出:
        features', cls_labels', boundary_labels' (若给出)
    """
    def __init__(self, swap_prob=0.5):
        self.swap_prob = swap_prob

    def __call__(self, input):
        
        features, cls_labels, boundary_labels = input[0], input[1], input[2] 
        # 确保为 numpy array
        x = features.numpy() if torch.is_tensor(features) else features.copy()
        y = cls_labels.numpy() if torch.is_tensor(cls_labels) else cls_labels.copy()

        T = len(y)
        # 找到分段边界
        segments = []  # list of (start, end) 半开区间
        start = 0
        for t in range(1, T):
            if y[t] != y[t-1]:
                segments.append((start, t))
                start = t
        segments.append((start, T))

        # 若分段数少于2，不交换
        if len(segments) < 2 or random.random() > self.swap_prob:
            return features, cls_labels, boundary_labels

        # 随机选择两个不同段
        i, j = random.sample(range(len(segments)), 2)
        (s1, e1), (s2, e2) = segments[i], segments[j]

        # 构造新的索引顺序
        order = list(range(T))
        # 第二段->占位
        part1 = order[s2:e2]
        part2 = order[s1:e1]
        if i < j:
            new_order = (
                order[:s1]
                + part1
                + order[e1:s2]
                + part2
                + order[e2:]
            )
        else:
            new_order = (
                order[:s2]
                + part2
                + order[e2:s1]
                + part1
                + order[e1:]
            )

        # 重排
        new_x = x[..., new_order] if x.ndim > 1 else x[new_order]
        new_y = y[new_order]
        if boundary_labels is not None:
            b = boundary_labels.numpy() if torch.is_tensor(boundary_labels) else boundary_labels.copy()
            new_b = b[new_order]
        else:
            new_b = None

        # 转回 tensor 如原
        if torch.is_tensor(features):
            new_x = torch.from_numpy(new_x)
        if torch.is_tensor(cls_labels):
            new_y = torch.from_numpy(new_y)
        if boundary_labels is not None and torch.is_tensor(boundary_labels):
            new_b = torch.from_numpy(new_b)

        return [new_x, new_y, new_b]

# 使用示例
if __name__ == '__main__':
    # 假设 features: [C, T]，cls_labels: [T]
    C, T = 3, 20
    feats = torch.randn(C, T)
    # 例子: labels 00000111222000033333
    labels = torch.tensor([0]*5 + [1]*3 + [2]*3 + [0]*5 + [3]*4)
    boundary_labels = torch.tensor([1 if labels[i] != labels[i-1] else 0 for i in range(1, T)])
    boundary_labels = torch.cat((torch.tensor([1]), boundary_labels))  # 第一个元素为1，表示第0帧是边界
    transform = SegmentSwapTransform()
    x2, y2, _ = transform(feats, labels, boundary_labels)
    print('原 labels:', labels.tolist())
    print('新 labels:', y2.tolist())
    print('原 features:', feats)
    print('新 features:', x2)
    print("原边界:", boundary_labels.tolist())
    print("new boundary:", _)

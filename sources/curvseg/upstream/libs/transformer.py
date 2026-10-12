import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TempDownSamp(object):
    def __init__(self, downsamp_rate: int = 1) -> None:
        super().__init__()
        self.downsamp_rate = downsamp_rate

    def __call__(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        feature, label, boundary = inputs[0], inputs[1], inputs[2]

        if self.downsamp_rate > 1:
            feature = feature[:, :, :: self.downsamp_rate]
            label = label[:: self.downsamp_rate]

            # not to remove boundary at odd number frames.
            idx = torch.where(boundary == 1.0)[0]
            boundary = torch.zeros(len(label))
            boundary[torch.div(idx, self.downsamp_rate, rounding_mode='trunc')] = 1.0

        return [feature, label, boundary]


class BoundarySmoothing(object):
    def __init__(self, kernel_size: int = 11) -> None:
        super().__init__()
        self.smoothing = GaussianSmoothing(kernel_size=kernel_size)

    def __call__(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        feature, label, boundary = inputs[0], inputs[1], inputs[2]

        boundary = boundary.view(1, 1, -1)
        boundary = self.smoothing(boundary)
        boundary = boundary.view(-1)

        return [feature, label, boundary]


class ToTensor(object):
    def __call__(self, inputs: List[np.ndarray]) -> List[torch.Tensor]:
        feature, label, boundary = inputs[0], inputs[1], inputs[2]

        # from numpy to tensor
        feature = torch.from_numpy(feature).float()
        label = torch.from_numpy(label).long()
        boundary = torch.from_numpy(boundary).float()

        # arrange feature and label in the temporal duration
        if feature.shape[2] == label.shape[0]:
            pass
        elif feature.shape[2] > label.shape[0]:
            feature = feature[:, : label.shape[0]]
        else:
            label = label[: feature.shape[2]]
            boundary = boundary[: feature.shape[2]]

        return [feature, label, boundary]


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d tensor.
    Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """
    """
    对1维张量应用高斯平滑。
    使用深度卷积分别对输入的每个通道进行滤波。
    参数:
        kernel_size (int): 高斯核的大小。
        sigma (float): 高斯核的标准差。
    """

    def __init__(self, kernel_size: int = 15, sigma: float = 1.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrid = torch.meshgrid(torch.arange(kernel_size))[0].float()

        mean = (kernel_size - 1) / 2
        kernel = kernel / (sigma * math.sqrt(2 * math.pi))
        kernel = kernel * torch.exp(-(((meshgrid - mean) / sigma) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        # kernel = kernel / torch.max(kernel)

        self.kernel = kernel.view(1, 1, *kernel.size())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        _, c, _ = inputs.shape
        inputs = F.pad(
            inputs,
            pad=((self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2),
            mode="reflect",
        )
        kernel = self.kernel.repeat(c, *[1] * (self.kernel.dim() - 1)).to(inputs.device)
        return F.conv1d(inputs, weight=kernel, groups=c)

import random
import math

import torch

class RandomJointMask:
    def __init__(self, mask_prob=0.1):
        """
        :param mask_prob: 每个关节点被遮挡的概率
        """
        self.mask_prob = mask_prob

    def __call__(self, input):
        skeleton, a, b = input[0], input[1], input[2]
        # skeleton: (T, V, C)
        mask = (torch.rand(skeleton.shape) > self.mask_prob).float()  # shape: (T, V)
        # mask = mask.unsqueeze(-1)  # shape: (T, V, 1)
        return [skeleton * mask, a, b]


class RandomNDRotation:
    """
    在任意两个维度组成的平面上进行随机旋转（N 维）
    """
    def __init__(self, angle_range=(-30, 30), num_rotations=1, degrees=True):
        self.angle_range = angle_range
        self.num_rotations = num_rotations
        self.degrees = degrees

    def __call__(self, input):
        # skeleton: Tensor of shape [T, V, C] or [V, C]
        skeleton, a, b = input[0], input[1], input[2]
        skeleton = skeleton.permute(1, 2, 0)
        
        *rest, C = skeleton.shape
        for _ in range(self.num_rotations):
            # 随机选取两个不同维度
            i, j = random.sample(range(C), 2)
            theta = random.uniform(*self.angle_range)
            if self.degrees:
                theta = math.radians(theta)
            cos_t, sin_t = math.cos(theta), math.sin(theta)

            # 提取并旋转
            # 把 [-1,C] 维展平操作，也支持前两维
            coords_i = skeleton[..., i].clone()
            coords_j = skeleton[..., j].clone()
            skeleton[..., i] = cos_t * coords_i - sin_t * coords_j
            skeleton[..., j] = sin_t * coords_i + cos_t * coords_j

        return [skeleton.permute(2, 0, 1), a, b]



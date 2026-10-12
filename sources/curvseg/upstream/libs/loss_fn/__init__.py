import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal import FocalLoss
from .tmse import TMSE, GaussianSimilarityTMSE
from .curve import CurvatureLoss
from .circle import CircleLoss, convert_label_to_similarity

__all__ = ["ActionSegmentationLoss", "BoundaryRegressionLoss"]


class ActionSegmentationLoss(nn.Module):
    """
    Loss Function for Action Segmentation
    You can choose the below loss functions and combine them.
        - Cross Entropy Loss (CE)
        - Focal Loss
        - Temporal MSE (TMSE)
        - Gaussian Similarity TMSE (GSTMSE)
        - Curvature Loss (CL)
    """
    def __init__(
        self,
        ce: bool = True,
        focal: bool = True,
        tmse: bool = False,
        gstmse: bool = False,
        cl: bool = False,
        circle: bool = False,
        weight: Optional[float] = None,
        threshold: float = 4,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        focal_weight: float = 1.0,
        tmse_weight: float = 0.15,
        gstmse_weight: float = 0.15,
        cl_weight: float = 1,
        circle_weight: float = 1
    ) -> None:
        super().__init__()
        self.criterions = []
        self.weights = []
        self.ignore_index = ignore_index

        if ce: #CrossEntropy
            self.criterions.append(
                nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            )
            self.weights.append(ce_weight)

        if focal: #Focal Loss
            self.criterions.append(FocalLoss(ignore_index=ignore_index))
            self.weights.append(focal_weight)

        if tmse: #TMSE（Topographic Map Similarity Enhancement）
            self.criterions.append(TMSE(threshold=threshold, ignore_index=ignore_index))
            self.weights.append(tmse_weight)

        if gstmse: # Gaussian Similarity TMSE
            self.criterions.append(
                GaussianSimilarityTMSE(threshold=threshold, ignore_index=ignore_index)
            )
            self.weights.append(gstmse_weight)

        if cl: # Curvature Loss
            self.criterions.append(CurvatureLoss(q=10, w=10))
            self.weights.append(cl_weight)

        if circle: # Circle Loss
            self.criterions.append(CircleLoss(m=0.25, gamma=256))
            self.weights.append(circle_weight)

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)
        print("Using the following loss functions:")
        for criterion, weight in zip(self.criterions, self.weights):
            print(f"{criterion.__class__.__name__} (weight: {weight:.2f})")

    def forward(
        self, preds: torch.Tensor, gts: torch.Tensor, sim_index: torch.Tensor, frame_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            preds: torch.float (N, C, T).
            gts: torch.long (N, T).
            sim_index: torch.float (N, C', T).
             frame_weights: torch.float (N, T) 或 None
        """

        loss = 0.0
        N, C, T = preds.shape
        for criterion, weight in zip(self.criterions, self.weights):
            # CrossEntropyLoss 特殊处理：支持逐帧加权
            if isinstance(criterion, nn.CrossEntropyLoss):
                if frame_weights is not None:
                    # flatten 时间维到 batch 维
                    pred_flat = preds.permute(0, 2, 1).reshape(-1, C)   # [N*T, C]
                    gt_flat   = gts.reshape(-1)                        # [N*T]
                    # 逐元素 CE
                    ce_flat = F.cross_entropy(
                        pred_flat,
                        gt_flat,
                        ignore_index=self.ignore_index,
                        weight=criterion.weight,
                        reduction='none'
                    )                                                   # [N*T]
                    # 对齐并加权
                    
                    N_w, T_w = frame_weights.shape
                    assert N_w == N, "batch size mismatch between preds and frame_weights"
                    if T_w != T:
                        # 插值到 T_pred 长度
                        # (N, T_w) -> (N, 1, T_w)
                        fw = frame_weights.unsqueeze(1).float()
                        # F.interpolate 要求输入是 float
                        fw = F.interpolate(fw, size=T, mode='linear', align_corners=False)
                        # -> (N, T_pred)
                        frame_weights = fw.squeeze(1)
                    weights_flat = frame_weights.reshape(-1)           # [N*T]
                    ce_loss = (ce_flat * weights_flat).mean()
                    loss += weight * ce_loss
                else:
                    # 默认行为：内部 reduction='mean'
                    loss += weight * criterion(preds, gts)
            elif isinstance(criterion, GaussianSimilarityTMSE):
                loss += weight * criterion(preds, gts, sim_index)
            elif isinstance(criterion, CircleLoss):
                inp_sp, inp_sn = convert_label_to_similarity(preds, gts)
                loss += weight * criterion(inp_sp, inp_sn)
            else:
                loss += weight * criterion(preds, gts)

        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import sys

class BoundaryRegressionLoss(nn.Module):
    """
    Boundary Regression Loss with Gaussian Smoothing
        bce: Binary Cross Entropy Loss for Boundary Prediction
        mse: Mean Squared Error
        focal: Focal Loss
        smoothed: Apply Gaussian Smoothing
    """

    def __init__(
        self,
        bce: bool = True,
        focal: bool = False,
        mse: bool = False,
        weight: Optional[float] = None,
        pos_weight: Optional[float] = None,
        smoothed: bool = False,  # 添加高斯平滑参数
        kernel_size: int = 5,    # 高斯平滑的核大小
        sigma: float = 1.0       # 高斯平滑的标准差
    ) -> None:
        super().__init__()

        self.criterions = []
        self.smoothed = smoothed
        self.kernel_size = kernel_size
        self.sigma = sigma

        if bce:
            self.criterions.append(
                nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
            )

        if focal:
            self.criterions.append(FocalLoss())

        if mse:
            self.criterions.append(nn.MSELoss())

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def apply_gaussian_smoothing(self, tensor: torch.Tensor) -> torch.Tensor:
        """对输入张量应用高斯平滑"""
        with torch.no_grad():
            # 生成高斯核
            k_size = self.kernel_size
            sigma = self.sigma
            x = torch.linspace(-sigma, sigma, k_size)
            gaussian = torch.exp(-x.pow(2) / (2 * sigma**2))
            gaussian = gaussian / gaussian.sum()

            # 转换成卷积核
            kernel = gaussian.view(1, 1, k_size).repeat(tensor.shape[1], 1, 1).to(tensor.device)

            # 应用高斯平滑
            smoothed_tensor = F.conv1d(
                F.pad(tensor, (k_size//2, k_size//2), mode='replicate'),
                kernel,
                groups=tensor.shape[1]
            )

            return smoothed_tensor

    def forward(self, preds: torch.Tensor, gts: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            preds: torch.float (N, 1, T).
            gts: torch. (N, 1, T).
            masks: torch.bool (N, 1, T).
        """
        if self.smoothed:
            # preds = self.apply_gaussian_smoothing(preds)
            gts = self.apply_gaussian_smoothing(gts)

        loss = 0.0
        batch_size = float(preds.shape[0])

        for criterion in self.criterions:
            for pred, gt, mask in zip(preds, gts, masks):
                loss += criterion(pred[mask], gt[mask])

        return loss / batch_size


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(size_average=True, reduce=True)):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
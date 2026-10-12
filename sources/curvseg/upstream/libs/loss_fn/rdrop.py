import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class RDroLoss(nn.Module):
    def __init__(self, constraint_type='kl'):
        super(RDroLoss, self).__init__()
        self.constraint_type = constraint_type

    def forward(self, logits1, logits2, mask):
        # 应用mask过滤元素
        logits1 = logits1[mask]
        logits2 = logits2[mask]

        if self.constraint_type == 'kl':
            # 双向KL散度约束
            kl_ab = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2.detach(), dim=-1), reduction='batchmean')
            kl_ba = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1.detach(), dim=-1), reduction='batchmean')
            loss = (kl_ab + kl_ba) / 2

        elif self.constraint_type == 'ce':
            # 交叉熵约束（硬标签）
            ce_ab = F.cross_entropy(logits1, logits2.detach().argmax(dim=-1))
            ce_ba = F.cross_entropy(logits2, logits1.detach().argmax(dim=-1))
            loss = (ce_ab + ce_ba) / 2

        elif self.constraint_type == 'l1':
            # 双向L1距离约束
            prob1_ab = F.softmax(logits1, dim=-1)
            prob2_ab = F.softmax(logits2.detach(), dim=-1)
            l1_ab = F.l1_loss(prob1_ab, prob2_ab, reduction='mean')

            prob2_ba = F.softmax(logits2, dim=-1)
            prob1_ba = F.softmax(logits1.detach(), dim=-1)
            l1_ba = F.l1_loss(prob2_ba, prob1_ba, reduction='mean')

            loss = (l1_ab + l1_ba) / 2

        else:
            raise ValueError(f"Unsupported constraint type: {self.constraint_type}")

        return loss
    

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossConstraintLoss(nn.Module):
    def __init__(self, constraint_type='kl', temperature=1.0):
        """
        初始化交叉约束损失模块
        
        参数:
            constraint_type (str): 约束类型，可选 'kl' (KL 散度), 'ce' (交叉熵), 'l1' (L1 损失)
            temperature (float): 温度参数，用于软化概率分布，默认为 1.0
        """
        super(CrossConstraintLoss, self).__init__()
        self.constraint_type = constraint_type
        self.temperature = temperature

    def forward(self, logits1, logits2, mask=None):
        """
        前向传播，计算两种预测分布之间的约束损失
        
        参数:
            logits1 (Tensor): 第一个模型的输出 logits，形状为 [batch_size, num_classes]
            logits2 (Tensor): 第二个模型的输出 logits，形状为 [batch_size, num_classes]
            mask (Tensor, optional): 用于筛选有效样本的掩码，形状为 [batch_size]，默认为 None
        
        返回:
            Tensor: 计算得到的约束损失
        """
        # 如果提供了掩码，筛选出有效样本
        if mask is not None:
            logits1 = logits1[mask]
            logits2 = logits2[mask]

        # 根据约束类型计算损失
        if self.constraint_type == 'kl':
            loss = self._kl_divergence(logits1, logits2)
        elif self.constraint_type == 'ce':
            loss = self._cross_entropy(logits1, logits2)
        elif self.constraint_type == 'l1':
            loss = self._l1_loss(logits1, logits2)
        else:
            raise ValueError(f"Unsupported constraint type: {self.constraint_type}")

        return loss

    def _kl_divergence(self, logits1, logits2):
        """计算双向 KL 散度的平均值"""
        # 计算两个方向的 KL 散度
        kl_ab = F.kl_div(
            F.log_softmax(logits1 / self.temperature, dim=-1),
            F.softmax(logits2.detach() / self.temperature, dim=-1),
            reduction='batchmean'
        )
        kl_ba = F.kl_div(
            F.log_softmax(logits2 / self.temperature, dim=-1),
            F.softmax(logits1.detach() / self.temperature, dim=-1),
            reduction='batchmean'
        )
        # 返回双向 KL 散度的平均值
        return (kl_ab + kl_ba) / 2

    def _cross_entropy(self, logits1, logits2):
        """计算交叉熵损失"""
        # 使用 logits2 作为 "标签" 来计算 logits1 的交叉熵
        ce_ab = F.cross_entropy(logits1, logits2.argmax(dim=-1))
        ce_ba = F.cross_entropy(logits2, logits1.argmax(dim=-1))
        # 返回双向交叉熵的平均值
        return (ce_ab + ce_ba) / 2

    def _l1_loss(self, logits1, logits2):
        """计算 L1 损失"""
        # 计算两个 logits 分布之间的 L1 距离
        l1_distance = torch.abs(logits1 - logits2).mean()
        return l1_distance
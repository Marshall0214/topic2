import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CurvatureLoss(nn.Module):
    def __init__(self, q, w=10):
        super().__init__()
        self.q = q
        self.w = w
    
    @staticmethod
    def _normalize_sequence(dist: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        对每个样本的距离序列做归一化到 [0,1]，形状 [B, T]
        """
        min_val = dist.min(dim=1, keepdim=True)[0]
        max_val = dist.max(dim=1, keepdim=True)[0]
        return (dist - min_val) / (max_val - min_val + eps)

    @staticmethod
    def _compute_step_distances(
        embs: torch.Tensor,
        step: int = 1,
        metric: str = 'euclidean'
    ) -> torch.Tensor:
        """
        计算特征序列 embs 在时序维度上每隔 step 的距离，形成长度 T 序列。

        Args:
            embs: [B, T, D]
            step: 时间间隔
            metric: 'euclidean' 或 'cosine'
        Returns:
            dist_seq: [B, T]，第一 step 个位置用 0 填充
        """
        B, T, D = embs.shape
        # 准备两端张量
        # 例如 step=1，则 x[:-1], x[1:]
        # pad 前面 step 个以保持输出长度
        if metric == 'euclidean':
            dist_fun = lambda x, y: torch.norm(x - y, dim=2)
        elif metric == 'cosine':
            # cos_distance = 1 - cos_sim
            dist_fun = lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=2)
        else:
            raise ValueError(f"Unsupported metric {metric}")

        # 构造对齐版本
        emb1 = embs[:, :-step, :]  # [B, T-step, D]
        emb2 = embs[:, step:, :]   # [B, T-step, D]
        dist = dist_fun(emb1, emb2)  # [B, T-step]

        # 在头尾 pad
        # 前 step 个设为 0
        pad_front = torch.zeros(B, step, device=embs.device)
        # 后面如果剩余，将尾部也填 0
        pad_back_len = T - pad_front.shape[1] - dist.shape[1]
        pad_back = torch.zeros(B, pad_back_len, device=embs.device)

        dist_seq = torch.cat([pad_front, dist, pad_back], dim=1)
        # 归一化
        dist_seq = CurvatureLoss._normalize_sequence(dist_seq)
        return dist_seq

    def curvature_estimation(self, embs, q=10, device=0, w=10, normalization=None):
        """
        Batch-optimized curvature estimation with PyTorch
        
        参数:
            embs: 输入的嵌入向量
            q: 量化参数，默认为10
            device: 计算设备，默认为0
            w: 窗口大小，默认为10
            normalization: 归一化方案，可选值为:
                None或'none' - 不归一化（默认）
                'temporal' - 时序归一化（沿seq_len维度）
                'channel' - 通道归一化（沿emb_dim维度）
                'global' - 除batch维度外全局归一化（沿emb_dim和seq_len维度）
        """
        embs = torch.tensor(embs, device=device)
        batch_size, emb_dim, seq_len = embs.shape
        
        # 根据选择的归一化方案进行处理
        if normalization is not None and normalization.lower() != 'none':
            # 计算归一化所需的范数
            if normalization.lower() == 'temporal':
                # 时序归一化：沿seq_len维度归一化
                norms = torch.norm(embs, p=2, dim=2, keepdim=True) + 1e-8
            elif normalization.lower() == 'channel':
                # 通道归一化：沿emb_dim维度归一化
                norms = torch.norm(embs, p=2, dim=1, keepdim=True) + 1e-8
            elif normalization.lower() == 'global':
                # 全局归一化：沿emb_dim和seq_len维度归一化
                norms = torch.norm(embs, p=2, dim=(1, 2), keepdim=True) + 1e-8
            else:
                raise ValueError(f"不支持的归一化方案: {normalization}，可选方案为'none', 'temporal', 'channel', 'global'")
            
            # 应用L2归一化
            embs = embs / norms
        q = self.q
        w = self.w
        
        # 转置维度用于卷积操作
        # embs = embs.permute(0, 2, 1)  # [B, D, T]
        
        # 构建左右padding
        embs_pad_left = F.pad(embs, (q-1, 0), mode='replicate')
        embs_pad_right = F.pad(embs, (0, q-1), mode='replicate')
        
        # 构建差分卷积核
        kernel = torch.zeros(emb_dim, 1, q, device=embs.device)
        kernel[:, :, 0] = -1
        kernel[:, :, -1] = 1
        
        # 并行计算左右变化向量
        cv_left = F.conv1d(embs_pad_left, kernel, groups=emb_dim)
        cv_right = F.conv1d(embs_pad_right, kernel, groups=emb_dim)
        
        # 恢复原始维度
        cv_left = cv_left.permute(0, 2, 1)  # [B, T, D]
        cv_right = cv_right.permute(0, 2, 1)
        
        # 计算向量范数
        norm_left = torch.norm(cv_left, dim=2, keepdim=True)  # [B, T, 1]
        norm_right = torch.norm(cv_right, dim=2, keepdim=True)
        
        # 计算余弦相似度
        cos_sim = F.cosine_similarity(cv_left, cv_right, dim=2)
        cos_sim = torch.clamp(cos_sim, -1+1e-6, 1-1e-6)
        
        # 计算角度和曲率
        angle = torch.acos(cos_sim)
        curvature = angle / (norm_left.squeeze() + norm_right.squeeze() + 1e-6)
        
        # 移动平均平滑
        curvature = curvature.unsqueeze(1)  # [B, 1, T]
        curvature_pad = F.pad(curvature, (w-1, w), mode='replicate')
        avg_kernel = torch.ones(1, 1, 2*w, device=embs.device) / (2*w)
        movavg = F.conv1d(curvature_pad, avg_kernel)
        
        # 反转并标准化到[0,1]
        movavg = movavg.squeeze(1)  # [B, T]
        movavg = movavg.max(dim=1, keepdim=True)[0] - movavg
        min_val = movavg.min(dim=1, keepdim=True)[0]
        max_val = movavg.max(dim=1, keepdim=True)[0]
        movavg = (movavg - min_val) / (max_val - min_val + 1e-6)

        
        curv = torch.squeeze(curvature)
        curv = torch.max(curv)-curv # make CP higher than In-Segment
        curv = (curv - torch.min(curv))/(torch.max(curv)-torch.min(curv))

        curv_reciprocal = ((norm_left.squeeze()+norm_right.squeeze())/(angle+1e-6))
        curv_reciprocal = (curv_reciprocal - torch.min(curv_reciprocal))/(torch.max(curv_reciprocal)-torch.min(curv_reciprocal))


        return curv, curv_reciprocal, movavg
    
    def forward(self, embs, labels):
        """
        Args:
            embs:  形状 [B, T, D] 的嵌入序列
            labels: 形状 [B, T] 的标签序列
        Returns:
            曲率损失值
        """
        # 计算预测曲率
        _, pred_curv, _ = self.curvature_estimation(embs)  # [B, T]
        
        # 生成目标曲率
        target = torch.zeros_like(pred_curv)
        target[:, 1:] = (labels[:, 1:] != labels[:, :-1]).float()
        print(target.shape, pred_curv.shape)
        torch.set_printoptions(profile="full")
        print(target[0], labels[0])
        exit()
        
        # 计算加权MSE损失
        return F.mse_loss(pred_curv, target)
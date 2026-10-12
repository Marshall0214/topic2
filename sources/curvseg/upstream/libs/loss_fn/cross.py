import torch
import torch.nn as nn
import torch.nn.functional as F


class RangeAdapter(nn.Module):
    def __init__(self, mode='sigmoid'):
        super().__init__()
        self.mode = mode
        
    def forward(self, logits):
        if self.mode == 'sigmoid':
            return torch.sigmoid(logits)  # 将logits2压缩到(0,1)
        elif self.mode == 'minmax':
            return (logits - logits.min()) / (logits.max() - logits.min() + 1e-8)
        else:
            return logits

def piecewise_sigmoid(x: torch.Tensor,
                      x0: float = 0.5,
                      k1: float = 10.0,
                      k2: float = 20.0) -> torch.Tensor:
    """
    分段 Sigmoid：
     - 当 x <= x0 时，使用陡峭度 k1；
     - 当 x >  x0 时，使用陡峭度 k2。
    保证 f(x0) = 0.5。
    
    Args:
        x  (Tensor): 输入张量，假设取值范围在 [0,1] 内；
        x0 (float): 分段点；
        k1 (float): 左半段（x<=x0）的陡峭度；
        k2 (float): 右半段（x> x0）的陡峭度；
    Returns:
        Tensor: 同形状的输出，映射到 (0,1)。
    """
    # 预分配输出
    y = torch.empty_like(x)
    
    # 左半段：x <= x0
    mask_l = x > x0
    if mask_l.any():
        # sigmoid(k1*(x - x0)) 在 x=x0 时的值是 0.5
        y[mask_l] = torch.sigmoid(k1 * (x[mask_l] - x0))
    
    # 右半段：x > x0
    mask_r = x <= x0
    if mask_r.any():
        y[mask_r] = torch.sigmoid(k2 * (x[mask_r] - x0))
    
    return y
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Beta

class TSConstraintLoss(nn.Module):
    def __init__(
        self,
        alpha=3.5,
        beta=3.5,
        # —— 新增映射相关参数 —— #
        map_method: str = 'None',      # 可选 ['sigmoid','tanh','piecewise_power','beta_cdf','piecewise_sigmoid']
        k: float = 7.0,                # sigmoid/tanh 的陡峭度
        shift: float = 0.2,            # 中心点 x0，默认靠近 0
        gamma: float = 2.0,            # piecewise_power 的幂指数
        beta_params: tuple = (2.0, 2.0),# beta_cdf 的 (alpha, beta)
        # —— 新增 loss_type 参数 —— #
        loss_type: str = 'mse'         # 可选 ['mse','l1','kl']
    ):
        super().__init__()
        self.alpha = alpha  
        self.beta  = beta   

        # —— 映射控制参数 —— #
        self.map_method = map_method
        self.k          = k
        self.shift      = shift
        self.gamma      = gamma
        self.beta_a, self.beta_b = beta_params
        self.loss_type  = loss_type.lower()  # 小写化，方便比较

        self.range_adapter = RangeAdapter()
        self.mse = nn.MSELoss()
        self.l1  = nn.L1Loss()

    def _remap(self, x: torch.Tensor) -> torch.Tensor:
        """
        对 [0,1] 范围内的 x 做非线性映射，
        方法由 self.map_method 决定。
        """
        if self.map_method == 'sigmoid':
            # f(x) = 1 / (1 + exp(-k * (x - x0)))
            return torch.sigmoid(self.k * (x - self.shift))

        elif self.map_method == 'tanh':
            # f(x) = (tanh(k*(x - x0)) + 1) / 2
            return 0.5 * (torch.tanh(self.k * (x - self.shift)) + 1.0)

        elif self.map_method == 'piecewise_power':
            # 分段幂：0–x0, x0–1 分别按幂函数拉升
            x0 = self.shift
            y = torch.empty_like(x)
            mask_l = x <= x0
            if mask_l.any():
                y[mask_l] = 0.5 * ((x[mask_l] / x0) ** self.gamma)
            mask_r = x > x0
            if mask_r.any():
                y[mask_r] = 0.5 + 0.5 * (((x[mask_r] - x0) / (1 - x0)) ** self.gamma)
            return y

        elif self.map_method == 'beta_cdf':
            # Beta 分布 CDF 映射
            beta_dist = Beta(self.beta_a, self.beta_b)
            return beta_dist.cdf(x)

        elif self.map_method == 'piecewise_sigmoid':
            return piecewise_sigmoid(x, self.shift, self.k, 200)

        elif self.map_method == 'None':
            # 不做任何映射
            return x

        else:
            raise ValueError(f"Unknown map_method '{self.map_method}'")

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        根据 self.loss_type 计算 pred vs target 的损失。
        pred 和 target 维度相同，都可视作在 [0,1] 区间内的密度/曲线。
        """
        if self.loss_type == 'mse':
            return self.mse(pred, target)

        elif self.loss_type == 'l1':
            return self.l1(pred, target)

        elif self.loss_type == 'kl':
            # 先避免 log(0)，将 p,q 裁在 [eps, 1]
            eps = 1e-6
            p = pred.clamp(min=eps, max=1.0)
            q = target.clamp(min=eps, max=1.0)
            # 归一化到“分布”——保证最后维度和为 1
            p_norm = p / p.sum(dim=-1, keepdim=True)
            q_norm = q / q.sum(dim=-1, keepdim=True)
            # KL(p||q) = sum p * log(p/q)
            return (p_norm * (p_norm.log() - q_norm.log())).sum(dim=-1).mean()

        else:
            raise ValueError(f"Unknown loss_type '{self.loss_type}'")

    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        mask: torch.Tensor,
        true_boundary: torch.Tensor = None
    ) -> torch.Tensor:
        """
        logits1: curvature，shape (..., T)
        logits2: boundary，shape (..., T)
        mask:    有效帧掩码，同样的 shape
        true_boundary: 可选，若提供，则对 logits1 添加辅助约束
        """
        # —— 1. 准备数据 —— #
        raw1 = logits1[mask]                      # 取出有效帧
        mapped1 = self._remap(raw1)               # remap 到 [0,1]（或其它映射）
        mapped2 = self.range_adapter(logits2[mask])  # boundary 用 RangeAdapter 压到 [0,1]

        # —— 2. 主约束：对称 Loss —— #
        # 先计算 mapped2 vs mapped1.detach，然后 mapped1 vs mapped2.detach
        loss1 = self._compute_loss(mapped2, mapped1.detach())
        loss2 = self._compute_loss(mapped1, mapped2.detach())
        main_loss = 0.5 * (loss1 + loss2)
        # main_loss = loss1

        # —— 3. 辅助约束：如果传入 true_boundary 就对 raw1 加损失 —— #
        aux_loss = torch.tensor(0.0, device=logits1.device)
        if true_boundary is not None:
            gt = true_boundary[mask].float()  # 假设已经在 [0,1]
            # 辅助 Loss 固定使用 L1 或 MSE
            if self.loss_type == 'l1':
                aux_loss = self.l1(raw1, gt)
            else:
                aux_loss = self.mse(raw1, gt)

        # —— 4. 最终加权返回 —— #
        return main_loss * self.alpha + aux_loss * self.beta

# import torch
# import torch.nn as nn
# import numpy as np
# from torch.distributions import Beta

# class TSConstraintLoss(nn.Module):
#     def __init__(
#         self,
#         alpha=3.5,
#         beta=3.5,
#         # —— 新增映射相关参数 —— #
#         map_method: str = 'None',      # 可选 ['sigmoid','tanh','piecewise_power','beta_cdf']
#         k: float = 7.0,                  # sigmoid/tanh 的陡峭度
#         shift: float = 0.2,               # 中心点 x0，默认靠近 0
#         gamma: float = 2.0,               # piecewise_power 的幂指数
#         beta_params: tuple = (2.0, 2.0),   # beta_cdf 的 (alpha, beta)
#     ):
#         super().__init__()
#         self.alpha = alpha  
#         self.beta  = beta   
#         # —— 映射控制参数 —— #
#         self.map_method  = map_method
#         self.k           = k
#         self.shift       = shift
#         self.gamma       = gamma
#         self.beta_a, self.beta_b = beta_params

#         self.range_adapter = RangeAdapter()
#         self.mse = nn.MSELoss()

#     def _remap(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         对 [0,1] 范围内的 x 做非线性映射，
#         方法由 self.map_method 决定。
#         """
#         if self.map_method == 'sigmoid':
#             # f(x) = 1 / (1 + exp(-k * (x - x0)))
#             return torch.sigmoid(self.k * (x - self.shift))

#         elif self.map_method == 'tanh':
#             # f(x) = (tanh(k*(x - x0)) + 1) / 2
#             return 0.5 * (torch.tanh(self.k * (x - self.shift)) + 1.0)

#         elif self.map_method == 'piecewise_power':
#             # 分段幂：0–x0, x0–1 分别扁平
#             # 注意：x0 约定在 (0,1) 之间
#             x0 = self.shift
#             y = torch.empty_like(x)
#             # 左半段
#             mask_l = x <= x0
#             if mask_l.any():
#                 y[mask_l] = 0.5 * ((x[mask_l] / x0) ** self.gamma)
#             # 右半段
#             mask_r = x > x0
#             if mask_r.any():
#                 y[mask_r] = 0.5 + 0.5 * (((x[mask_r] - x0) / (1 - x0)) ** self.gamma)
#             return y

#         elif self.map_method == 'beta_cdf':
#             # Beta 分布 CDF
#             # torch.distributions.Beta 有 cdf 方法
#             beta_dist = Beta(self.beta_a, self.beta_b)
#             return beta_dist.cdf(x)
#         elif self.map_method == 'piecewise_sigmoid':
#             return piecewise_sigmoid(x, self.shift, self.k, 200)
#         elif self.map_method == 'None':
#             # 无映射，直接返回原值
#             return x
#         else:
#             raise ValueError(f"Unknown map_method '{self.map_method}'")

#     def forward(
#         self,
#         logits1: torch.Tensor,
#         logits2: torch.Tensor,
#         mask: torch.Tensor,
#         true_boundary: torch.Tensor = None
#     ) -> torch.Tensor:
#         """
#         logits1: curvature, shape (..., T)
#         logits2: boundary，shape (..., T)
#         mask:    有效帧掩码，same shape as logits1/logits2
#         true_boundary: 可选，若提供，则用来约束 logits1
#         """
#         # —— 1. 准备数据 —— #
#         # 原始 logits1 先截取有效区间，再做 remap
#         raw1 = logits1[mask]
#         mapped1 = self._remap(raw1)

#         # logits2 经 RangeAdapter 压缩到 (0,1)
#         mapped2 = self.range_adapter(logits2[mask])

#         # —— 2. 主约束：时序位置敏感 MSE —— #
#         # 用 remap 后的 mapped1 作为“目标曲线”
#         main_loss = torch.mean((mapped2 - mapped1.detach())**2)
#         # —— 2. 如果提供了真值，就给 logits1 一个辅助约束 —— #
#         aux_loss = 0.0
#         if true_boundary is not None:
#             gt = true_boundary[mask].float()            # 真值边界（假设已在 [0,1]）
#             # 直接对 mapped1 与 gt 做 MSE
#             aux_loss = self.mse(raw1, gt)
#         main_loss = main_loss + torch.mean((mapped2.detach() - mapped1)**2) 
#         main_loss = main_loss / 2

#         # —— 3. （可选）辅助约束：平滑性 —— #
#         # time_diff = mapped2[1:] - mapped2[:-1]
#         # smooth_loss = torch.mean(torch.abs(time_diff))

#         # —— 4. 动态权重调整（示例） —— #
#         # if main_loss.item() > 0.5:
#         #     self.alpha, self.beta = 0.9, 0.1

#         return main_loss * self.alpha + aux_loss * self.beta

    
    


class RobustTSConstraint(nn.Module):
    def __init__(self, 
                 weight = 3.0,
                 range_mode='sigmoid', 
                 temporal_weight=4.0, 
                 smooth_weight=0.3):
        super().__init__()
        self.weight = weight
        self.range_adapter = RangeAdapter(range_mode)
        self.temporal_weight = temporal_weight  # 过渡点权重系数
        self.smooth_weight = smooth_weight       # 平滑约束强度
        
    def forward(self, logits1, logits2, mask):
        # 数据预处理
        logits1 = logits1[mask]         # 固定目标分布
        logits2 = self.range_adapter(logits2[mask])  # 范围对齐
        
        # 核心L1约束（带时序敏感权重）
        weight_matrix = 1 + self.temporal_weight * logits1  # 过渡点强化
        l1_loss = torch.mean(torch.abs(logits2 - logits1.detach()))
        l1_loss = l1_loss + torch.mean(torch.abs(logits2.detach() - logits1))
        l1_loss = l1_loss / 2.0
        
        # 改进的平滑约束（双向差分）
        forward_diff = torch.abs(logits2[1:] - logits2[:-1])
        backward_diff = torch.abs(logits2[:-1] - logits2[1:]) 
        smooth_loss = 0.5 * (forward_diff.mean() + backward_diff.mean())
        
        # return l1_loss + self.smooth_weight * smooth_loss
        return l1_loss * self.weight
    
class KLTSConstraint(nn.Module):
    def __init__(self, 
                 temp_control=0.5,
                 smooth_weight=0.3,
                 range_mode='sigmoid'):
        super().__init__()
        self.weight = 5.0
        self.temperature = nn.Parameter(torch.tensor(temp_control))  # 可学习温度系数
        self.smooth_weight = smooth_weight
        self.range_adapter = RangeAdapter(range_mode)  # 使用统一的范围适配器

    def forward(self, logits1, logits2, mask):
        # 数据预处理（保持双向梯度）
        logits1 = self.range_adapter(logits1[mask])  # logits1也进行范围对齐
        logits2 = self.range_adapter(logits2[mask])
        
        # 双向KL约束（无梯度阻断）
        kl_loss = self._bidirectional_kl(logits1, logits2)
        
        # 时序敏感加权
        transition_mask = (logits1 > 0.7).float().detach()  # 仅用数值，不参与梯度
        weighted_loss = (1 + 3*transition_mask) * kl_loss
        main_loss = weighted_loss.mean()
        
        # 双向平滑约束
        smooth_loss = self._temporal_smoothness(logits1, logits2)
        
        return main_loss * self.weight

    def _bidirectional_kl(self, logits1, logits2):
        """双向KL散度计算"""
        prob1 = F.softmax(logits1 / self.temperature.abs(), dim=-1)
        prob2 = F.softmax(logits2 / self.temperature.abs(), dim=-1)
        
        kl_ab = F.kl_div(
            F.log_softmax(logits1, dim=-1),
            prob2.detach(),  # 阻断反向传播防止循环依赖
            reduction='none'
        ).sum(-1)
        
        kl_ba = F.kl_div(
            F.log_softmax(logits2, dim=-1),
            prob1.detach(),  # 同样阻断反向传播
            reduction='none'
        ).sum(-1)
        
        return (kl_ab + kl_ba) / 2

    def _temporal_smoothness(self, logits1, logits2):
        """双向时序平滑约束"""
        # logits1的平滑约束
        diff1 = torch.abs(logits1[1:] - logits1[:-1])
        smooth1 = diff1.mean()
        
        # logits2的平滑约束
        diff2 = torch.abs(logits2[1:] - logits2[:-1])
        smooth2 = diff2.mean()
        
        return (smooth1 + smooth2) / 2

    def get_temperature(self):
        """获取当前有效温度值"""
        return self.temperature.abs().item()
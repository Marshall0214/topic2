from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import copy
import math
import numpy as np

from .tcn import SingleStageTCN
from .SP import MultiScale_GraphConv
import torch.nn.functional as F

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class Linear_Attention(nn.Module):
    def __init__(self,
                 in_channel,
                 n_features,
                 out_channel,
                 n_heads=4,
                 drop_out=0.05
                 ):
        super().__init__()
        self.n_heads = n_heads

        self.query_projection = nn.Linear(in_channel, n_features)
        self.key_projection   = nn.Linear(in_channel, n_features)
        self.value_projection = nn.Linear(in_channel, n_features)
        self.out_projection   = nn.Linear(n_features, out_channel)
        self.dropout          = nn.Dropout(drop_out)

    def elu(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, mask, weight=None):
        # —— 尺寸检查 —— #
        # queries: [B, L, in_channel]
        assert queries.dim() == 3, f"queries must be 3D, got {queries.shape}"
        B, L, Cq = queries.shape

        # keys, values: [B, S, in_channel]
        assert keys.dim() == 3 and values.dim() == 3, \
            f"keys and values must be 3D, got {keys.shape}, {values.shape}"
        Bk, S, Ck = keys.shape
        Bv, Sv, Cv = values.shape
        assert Bk == B and Bv == B, "batch size mismatch between queries/keys/values"
        assert Cq == Ck == Cv, "feature dim mismatch between queries/keys/values"
        assert Sv == S, "length mismatch between keys and values"

        # mask: [B, 1, L]
        assert mask.dim() == 3 and mask.shape[0] == B and mask.shape[2] == L, \
            f"mask must be [B,1,L], got {mask.shape}"

        # weight: optional [B, L]
        if weight is not None:
            if weight.dim() != 2:
                weight = weight.view(B, L)  # 尝试将 weight 转换为 [B, L]
            assert weight.dim() == 2 and weight.shape == (B, L), \
                f"when provided, weight must be [B,L], got {weight.shape}"

        # —— 原有流程 —— #
        # 1) 线性映射 + 拆头
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys    = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values  = self.value_projection(values).view(B, S, self.n_heads, -1)

        queries = queries.transpose(1, 2)  # [B, heads, L, C']
        keys    = keys.transpose(1, 2)     # [B, heads, S, C']
        values  = values.transpose(1, 2)   # [B, heads, S, C']

        # 2) 激活
        queries = self.elu(queries)
        keys    = self.elu(keys)

        # 3) 先验矩阵 KV
        KV = torch.einsum('bhsc,bhsv->bhcv', keys, values)  # [B, heads, C', C']

        # 4) 归一化因子 Z
        Z = 1.0 / (torch.einsum('bhlc,bhc->bhl', queries, keys.sum(dim=2)) + 1e-6)  # [B, heads, L]

        # 5) 可选权重调整
        if weight is not None:
            Z = Z * weight.unsqueeze(1)  # broadcast to [B, heads, L]

        # 6) 输出组合
        x = torch.einsum('bhcv,bhlc,bhl->bhlv', KV, queries, Z)  # [B, heads, L, C']

        # 7) 还原多头 & 投影
        x = x.transpose(1, 2).reshape(B, L, -1)  # [B, L, heads*C']
        x = self.out_projection(x)
        x = self.dropout(x)

        # 8) 掩码
        return x * mask[:, 0, :, None]



# class Linear_Attention(nn.Module):
#     def __init__(self,
#                  in_channel,
#                  n_features,
#                  out_channel,
#                  n_heads=4,
#                  drop_out=0.05
#                  ):
#         super().__init__()
#         self.n_heads = n_heads

#         self.query_projection = nn.Linear(in_channel, n_features)
#         self.key_projection = nn.Linear(in_channel, n_features)
#         self.value_projection = nn.Linear(in_channel, n_features)
#         self.out_projection = nn.Linear(n_features, out_channel)
#         self.dropout = nn.Dropout(drop_out) #0.05

#     def elu(self, x):
#         return torch.sigmoid(x)
#         # return torch.nn.functional.elu(x) + 1
        
#     def forward(self, queries, keys, values, mask):

#         B, L, _ = queries.shape
#         _, S, _ = keys.shape
#         queries = self.query_projection(queries).view(B, L, self.n_heads, -1) 
#         keys = self.key_projection(keys).view(B, S, self.n_heads, -1)         
#         values = self.value_projection(values).view(B, S, self.n_heads, -1)   
        
#         queries = queries.transpose(1, 2)
#         keys = keys.transpose(1, 2)
#         values = values.transpose(1, 2) #（n,head,t,c）

#         queries = self.elu(queries)
#         keys = self.elu(keys)
#         KV = torch.einsum('...sd,...se->...de', keys, values) # （n,head,t,c）,（n,head,t,c）->(n,head,c,c)
#         Z = 1.0 / torch.einsum('...sd,...d->...s',queries, keys.sum(dim=-2)+1e-6) #（n,head,t,c）,（n,head,c） ->(n,head,t)

#         x = torch.einsum('...de,...sd,...s->...se', KV, queries, Z).transpose(1, 2) #(n,head,c,c),(n,head,t,c),(n,head,t)->(n,head,t,c)

#         x = x.reshape(B, L, -1) #4 head（n,t,c）
#         x = self.out_projection(x)
#         x = self.dropout(x) #0.05

#         return x * mask[:, 0, :, None]

class AttModule(nn.Module):
    def __init__(self, dilation, in_channel, out_channel, stage, alpha):
        super(AttModule, self).__init__()
        self.stage = stage
        self.alpha = alpha

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
            ) #膨胀卷积
        # self.feed_forward_ = nn.Sequential(
        #     nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation),
        #     nn.ReLU()
        #     ) #膨胀卷积
        
        self.instance_norm = nn.InstanceNorm1d(out_channel, track_running_stats=False)
        # self.instance_norm_ = nn.InstanceNorm1d(out_channel, track_running_stats=False)
        self.att_layer = Linear_Attention(out_channel, out_channel, out_channel)
        
        self.conv_out = nn.Conv1d(out_channel, out_channel, 1)
        self.dropout = nn.Dropout()
        
    def forward(self, x, f, mask):

        out = self.feed_forward(x)
        # out_f = self.feed_forward_(f) if f is not None else None
        
        if self.stage == 'encoder':
            q = self.instance_norm(out).permute(0, 2, 1)
            if f is not None:
                f = f.permute(0, 2, 1)
                # print(f.shape, q.shape)
                out = self.alpha * self.att_layer(q, q, f, mask).permute(0, 2, 1) + out
            else:
                out = self.alpha * self.att_layer(q, q, q, mask).permute(0, 2, 1) + out
            
        else:
            assert f is not None
            q = self.instance_norm(out).permute(0, 2, 1)
            f = f.permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, f, mask).permute(0, 2, 1) + out
       
        out = self.conv_out(out)
        
        out = self.dropout(out)
        # print(mask.shape)
        return (x + out) * mask

class SFI(nn.Module):
    def __init__(self, in_channel, n_features):
        super().__init__()
        self.conv_s = nn.Conv1d(in_channel, n_features, 1) #19->64
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Linear(n_features, n_features),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(n_features, n_features)) #64—>64
        
    def forward(self, feature_s, feature_t, mask):
        # Feature_s comes from space (n, t, v) feature_t comes from the previous layer of time (n, t, c)
        feature_s = feature_s.permute(0, 2, 1) #(n,v,t)
        n, c, t = feature_s.shape
        feature_s = self.conv_s(feature_s) #(n,v,t)->(n,c,t)
        map = self.softmax(torch.einsum("nct,ndt->ncd", feature_s, feature_t)/t) #（n,c,c）
        feature_cross = torch.einsum("ncd,ndt->nct", map, feature_t) #（n,c,c),（n,c,t）->（n,c,t）
        feature_cross = feature_cross + feature_t
        feature_cross = feature_cross.permute(0, 2, 1) #(n,t,c）
        feature_cross = self.ff(feature_cross).permute(0, 2, 1) + feature_t

        return feature_cross * mask

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class STI(nn.Module):
#     def __init__(
#         self, node, in_channel, n_features, out_channel,
#         num_layers, SFI_layer,
#         M=2,                    # 每帧要预测 M 个高斯
#         channel_masking_rate=0.3,
#         alpha=1, div_margin=0.1, div_weight=0.1,
#         pool_len=64,            # 将原始 t_len 池化到 64 帧
#         use_gaussian_weighting=True, use_diversity_loss=True
#     ):
#         super().__init__()
#         self.SFI_layer = SFI_layer
#         self.Shared_layer = [0, 1, 2, 3, 4, 5, 6]
#         self.L = len(SFI_layer) + 1
#         self.channel_masking_rate = channel_masking_rate
#         self.dropout = nn.Dropout2d(p=channel_masking_rate)

#         self.use_gaussian_weighting = use_gaussian_weighting
#         self.use_diversity_loss = use_diversity_loss

#         # 初始化卷积
#         self.conv_in = nn.Conv2d(in_channel, self.L, kernel_size=1)
#         self.conv_t  = nn.Conv1d(node, n_features, 1)

#         # SFI 融合层，与原来一致
#         self.SFI_layers = nn.ModuleList([SFI(node, n_features) for _ in range(self.L-1)])
#         self.SFI_layers_c = nn.ModuleList([SFI(node, n_features) for _ in range(self.L-1)])
#         self.SFI_layers_l = nn.ModuleList([SFI(node, n_features) for _ in range(self.L-1)])

#         # 多层 Attention 模块，与原来一致
#         self.layers = nn.ModuleList([AttModule(2**i, n_features, n_features, 'encoder', alpha)
#                                         for i in range(num_layers)])
#         self.layers_c = nn.ModuleList([AttModule(2**i, n_features, n_features, 'encoder', alpha)
#                                        for i in range(num_layers)])
#         self.layers_l = nn.ModuleList([AttModule(2**i, n_features, n_features, 'encoder', alpha)
#                                        for i in range(num_layers)])

#         # --------- 调整后的高斯混合参数部分 ---------
#         # M 表示“每帧预测的高斯数量”，pool_len 表示池化后帧数（如 64）
#         self.M = M
#         self.pool_len = pool_len      # 先把原始 t_len 池化到 pool_len 帧
#         self.G = pool_len * M         # 总共要预测的高斯数 G = 64 × M
#         self.eps = 1e-6
#         self.div_margin = div_margin
#         self.div_weight = div_weight

#         # 为 diversity loss 做掩码：在所有 G 个高斯之间计算 pairwise 时，用于去掉对角线
#         eye = torch.eye(self.G, dtype=torch.bool)
#         self.register_buffer('div_mask', ~eye)

#         # 在 forward 里第一次需要时再初始化下面这两个 ModuleList
#         # gauss_cls_pred: 用于“预测 μ_raw”  -> 对应每层的 [B, C, pool_len] -> [B, 2*M, pool_len]
#         # gauss_loc_pred: 用于“预测 logσ_raw”
#         self.gauss_cls_pred = None
#         self.gauss_loc_pred = None

#         # 用来保存每层的中间高斯信息（仅作可视化或 debug 用）
#         self.gauss_info = {
#             'mu_c': [None] * num_layers,
#             'sigma_c': [None] * num_layers,
#             'g_c': [None] * num_layers,
#             'scale_c': [None] * num_layers,
#             'w_c': [None] * num_layers,
#             'mu_l': [None] * num_layers,
#             'sigma_l': [None] * num_layers,
#             'g_l': [None] * num_layers,
#             'scale_l': [None] * num_layers,
#             'w_l': [None] * num_layers
#         }

#         # “每个高斯对应的 scale” 预测器：输入维度会是 (n_features × G)，输出 G 个 scale
#         hidden = n_features * self.G // 2
#         self.scale_mlp_c = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(n_features * self.G, hidden),
#                 nn.ReLU(),
#                 nn.Linear(hidden, self.G),
#                 nn.Softplus()
#             ) for _ in range(num_layers)
#         ])
#         self.scale_mlp_l = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(n_features * self.G, hidden),
#                 nn.ReLU(),
#                 nn.Linear(hidden, self.G),
#                 nn.Softplus()
#             ) for _ in range(num_layers)
#         ])

#         # ===== 多层特征融合时不再用 conv1d 累加，而是网络预测权重 =====
#         self.layer_weight_net_c = nn.Sequential(
#             nn.Linear(n_features * num_layers, n_features),
#             nn.ReLU(),
#             nn.Linear(n_features, num_layers),
#             nn.Softmax(dim=1)
#         )
#         self.layer_weight_net_l = nn.Sequential(
#             nn.Linear(n_features * num_layers, n_features),
#             nn.ReLU(),
#             nn.Linear(n_features, num_layers),
#             nn.Softmax(dim=1)
#         )

#         # 最后输出分支
#         self.conv_out_c = nn.Conv1d(n_features, out_channel, 1)
#         self.conv_out_l = nn.Conv1d(n_features, out_channel, 1)

#     def _init_gauss_predictors(self, feat_dim):
#         """
#         初始化两个 ModuleList，按层数分别为 gauss_cls_pred（C->2M）和 gauss_loc_pred（C->2M）。
#         这里用 Conv1d 而不是 Linear，因为要对 pool 后的 [B, C, pool_len] 进行“逐帧”预测：
#           - 输入：pool_feat_c->[B, C, pool_len]
#           - gauss_cls_pred[idx](pool_feat_c) -> [B, 2*M, pool_len]
#           - split 出 mu_raw_unscaled 与 logvar_raw_unscaled，均为 [B, M, pool_len]
#         """
#         self.gauss_cls_pred = nn.ModuleList([
#             nn.Conv1d(feat_dim, 2 * self.M, kernel_size=1)  # 逐帧预测 2*M 个数
#             for _ in range(self.L)
#         ])
#         self.gauss_loc_pred = nn.ModuleList([
#             nn.Conv1d(feat_dim, 2 * self.M, kernel_size=1)
#             for _ in range(self.L)
#         ])
#         self.add_module('gauss_cls_pred', self.gauss_cls_pred)
#         self.add_module('gauss_loc_pred', self.gauss_loc_pred)

#     def forward(self, x, mask, joint_text_embedding=None):
#         """
#         x: [B, in_channel, node, t_len]
#         mask: [B, 1, t_len]
#         """
#         # 1. channel dropout
#         if self.channel_masking_rate > 0:
#             x = self.dropout(x)

#         # 2. split feat_s / feat_t，并做最初映射
#         x = self.conv_in(x)                  # [B, L, node, t_len]
#         feat_s, feat_t = torch.split(x, (self.L-1, 1), dim=1)
#         feat_t = feat_t.squeeze(1).permute(0, 2, 1)  # [B, node, t_len]
#         feature_st = self.conv_t(feat_t)            # [B, C, t_len]
#         B, C, t_len = feature_st.shape

#         # 3. 第一次进入时，如果需要高斯加权或多样性损失，提前初始化预测器
#         if (self.use_gaussian_weighting or self.use_diversity_loss) and (self.gauss_cls_pred is None):
#             # pool 到 64 帧：feat_dim = C（因为我们用 Conv1d 逐帧预测，所以只需 C 作为输入通道数）
#             feat_dim = C
#             self._init_gauss_predictors(feat_dim)
#             for m in self.gauss_cls_pred: m.to(x.device)
#             for m in self.gauss_loc_pred: m.to(x.device)

#         div_loss = 0.0
        
#         all_feats_c, all_feats_l = [feature_st], [feature_st]

#         feature_st_c, feature_st_l = feature_st, feature_st

#         for idx, (layer_c, layer_l) in enumerate(zip(self.layers_c, self.layers_l)):
#             # 4.1 如果在 SFI 层，则先做跨模态融合
#             weights_cls, weights_loc = [], []
#             if idx in self.SFI_layer:
#                 if idx in self.Shared_layer:
#                     sfi_idx = self.SFI_layer.index(idx)
#                     feature_st = self.SFI_layers[sfi_idx](feat_s[:, sfi_idx], feature_st, mask)
#                     feature_st_c, feature_st_l = feature_st, feature_st
#                 else:
#                     sfi_idx = self.SFI_layer.index(idx)
#                     feature_st_c = self.SFI_layers_c[sfi_idx](feat_s[:, sfi_idx], feature_st_c, mask)
#                     feature_st_l = self.SFI_layers_l[sfi_idx](feat_s[:, sfi_idx], feature_st_l, mask)
#             if idx in self.Shared_layer:
#                 feature_st = self.layers[idx](feature_st, None, mask)
#                 feature_st_c, feature_st_l = feature_st, feature_st
#                 all_feats_c.append(feature_st_c)
#                 all_feats_l.append(feature_st_l)
#             else:
#                 # 4.2 计算第 idx 层对应的 64 帧 × M 高斯
#                 if self.use_gaussian_weighting or self.use_diversity_loss:
#                     # 4.2.1 先把 feature_st_c 池化到 [B, C, pool_len=64]
#                     # feature_st_c_ = layer_c.instance_norm(layer_c.feed_forward(feature_st_c))  # [B, C, t_len]
#                     pooled_c = F.adaptive_avg_pool1d(feature_st_c.contiguous(), self.pool_len)  # [B, C, 64]
#                     # 用 Conv1d 逐帧预测 2*M 个参数
#                     # params_c_raw: [B, 2*M, 64]
#                     params_c_raw = self.gauss_cls_pred[idx](pooled_c)  # [B, 2*M, 64]
#                     # 拆成 mu_raw_unscaled 与 logvar_raw_unscaled（都形状 [B, M, 64]）
#                     mu_c_raw_unscaled, logvar_c_raw_unscaled = params_c_raw.chunk(2, dim=1)  # 各 [B, M, 64]

#                     # 对 logvar 做 softplus 确保正数
#                     sigma_c_raw_unscaled = torch.sigmoid(logvar_c_raw_unscaled)  # [B, M, 64]
#                     # 对 mu_raw 做 sigmoid，把值归一到 (0,1) 区间内
#                     mu_c_unit = torch.sigmoid(mu_c_raw_unscaled)  # [B, M, 64]

#                     # 4.2.2 多样性损失：对这 64×M 个 μ_raw_unscaled 进行 pairwise 约束
#                     if self.use_diversity_loss:
#                         mu_flat = mu_c_raw_unscaled.view(B, self.G)  # [B, 64*M]
#                         diff = torch.relu(self.div_margin - (mu_flat.unsqueeze(2) - mu_flat.unsqueeze(1)).abs())
#                         div_loss += diff[:, self.div_mask].mean()

#                     # 4.2.3 计算全局坐标下的 mu_abs 与 sigma_abs
#                     # pool_idxs 必须是 [1, 64, 1]，才能和 mu_c_unit.permute(0,2,1) ([B,64,M]) 广播对齐
#                     device = x.device
#                     pool_idxs = torch.arange(self.pool_len, device=device).view(1, self.pool_len, 1)  # [1, 64, 1]

#                     seg_len = t_len / float(self.pool_len)  # 每段的实际帧数

#                     # permute mu_c_unit 从 [B, M, 64] -> [B, 64, M]
#                     mu_c_unit = mu_c_unit.permute(0, 2, 1)       # [B, 64, M]
#                     sigma_c_unit = sigma_c_raw_unscaled.permute(0, 2, 1)  # [B, 64, M]

#                     # 先平移（broadcast 到 [B,64,M] 加上 pool_idxs：[1,64,1]），再放缩
#                     # print("pool_idxs shape: ", pool_idxs.shape, "mu_c_unit shape: ", mu_c_unit.shape)
#                     # exit()
#                     mu_c_abs = pool_idxs + mu_c_unit   # [B, 64, M]
#                     sigma_c_abs = sigma_c_unit          # [B, 64, M]    

#                     # 将 [B,64,M] reshape/flatten 为 [B, G, 1]（G = 64*M），方便后续计算
#                     mu_c_abs = mu_c_abs.reshape(B, self.G, 1)      # [B, G, 1]
#                     sigma_c_abs = sigma_c_abs.reshape(B, self.G, 1)# [B, G, 1]

#                     # time_idx: [1, 1, t_len]
#                     # time_idx = torch.linspace(0, self.pool_len, t_len, device=device).view(1, 1, t_len)  # [1,1,t_len]
#                     time_idx = torch.linspace(0, 1, t_len, device=device).view(1,1,t_len)
#                     time_idx = time_idx * self.pool_len   # 变成 [0,63]

#                     # time_idx = time_idx * (self.pool_len - 1)  # 归一到 [0,63]，形状保持 [1,1,t_len]

#                     # 4.2.4 计算高斯权重 g_c: [B, G, t_len]
#                     g_c = torch.exp(- (time_idx - mu_c_abs)**2 / (2 * sigma_c_abs**2 + self.eps))  # [B, G, t_len]

#                     # 4.2.5 计算与 feature_st_c 的加权特征并预测 scale_c
#                     # feature_st_c: [B, C, t_len], g_c: [B, G, t_len]
#                     fg_c_raw = torch.einsum('bct,bgt->btcg', feature_st_c, g_c).reshape(B * t_len, C * self.G)
#                     scale_c = self.scale_mlp_c[idx](fg_c_raw).view(B, self.G, t_len)  # [B, G, t_len]

#                     # 4.2.6 w_c: [B, t_len] = sum over G of (g_c * scale_c)
#                     w_c = torch.einsum('bgt,bgt->bt', g_c, scale_c)  # [B, t_len]
#                     w_c = w_c         # 保证在 [0.1, 1.0] 之间
#                     weights_cls.append(w_c)
#                     # print("mu_c_abs.min/max:", mu_c_abs.min().item(), mu_c_abs.max().item())
#                     # print("time_idx.min/max:", time_idx.min().item(), time_idx.max().item())
#                     # print("g_c.min/max:", g_c.min().item(), g_c.max().item())


#                     # 4.2.7 对 feature_st_l 做同样的操作
#                     # feature_st_l_ = layer_l.instance_norm(layer_l.feed_forward(feature_st_l))  # [B, C, t_len]
#                     pooled_l = F.adaptive_avg_pool1d(feature_st_l.contiguous(), self.pool_len)  # [B, C, 64]
#                     params_l_raw = self.gauss_loc_pred[idx](pooled_l)  # [B, 2*M, 64]
#                     mu_l_raw_unscaled, logvar_l_raw_unscaled = params_l_raw.chunk(2, dim=1)  # 各 [B, M, 64]
#                     sigma_l_raw_unscaled = torch.sigmoid(logvar_l_raw_unscaled)       # [B, M, 64]
#                     mu_l_unit = torch.sigmoid(mu_l_raw_unscaled)                             # [B, M, 64]

#                     if self.use_diversity_loss:
#                         mu_l_flat = mu_l_raw_unscaled.view(B, self.G)  # [B, G]
#                         diff2 = torch.relu(self.div_margin - (mu_l_flat.unsqueeze(2) - mu_l_flat.unsqueeze(1)).abs())
#                         div_loss += diff2[:, self.div_mask].mean()

#                     mu_l_unit = mu_l_unit.permute(0, 2, 1)        # [B, 64, M]
#                     sigma_l_unit = sigma_l_raw_unscaled.permute(0, 2, 1)  # [B,64,M]

#                     mu_l_abs = pool_idxs + mu_l_unit  # [B, 64, M]
#                     sigma_l_abs = sigma_l_unit          # [B, 64, M]    

#                     mu_l_abs = mu_l_abs.reshape(B, self.G, 1)     # [B, G, 1]
#                     sigma_l_abs = sigma_l_abs.reshape(B, self.G, 1)  # [B, G, 1]

#                     g_l = torch.exp(- (time_idx - mu_l_abs)**2 / (2 * sigma_l_abs**2 + self.eps))  # [B, G, t_len]

#                     fg_l_raw = torch.einsum('bct,bgt->btcg', feature_st_l, g_l).reshape(B * t_len, C * self.G)
#                     scale_l = self.scale_mlp_l[idx](fg_l_raw).view(B, self.G, t_len)  # [B, G, t_len]
#                     w_l = torch.einsum('bgt,bgt->bt', g_l, scale_l)  # [B, t_len]
#                     w_l = torch.clamp(w_l, min=0.1, max=1.0)        # [B, t_len]
#                     weights_loc.append(w_l)

#                     # 4.2.8 存储调试/可视化用的高斯信息
#                     self.gauss_info['mu_c'][idx] = mu_c_abs.detach().cpu().clone()
#                     self.gauss_info['sigma_c'][idx] = sigma_c_abs.detach().cpu().clone()
#                     self.gauss_info['g_c'][idx] = g_c.detach().cpu().clone()
#                     self.gauss_info['scale_c'][idx] = scale_c.detach().cpu().clone()
#                     self.gauss_info['w_c'][idx] = w_c.detach().cpu().clone()

#                     self.gauss_info['mu_l'][idx] = mu_l_abs.detach().cpu().clone()
#                     self.gauss_info['sigma_l'][idx] = sigma_l_abs.detach().cpu().clone()
#                     self.gauss_info['g_l'][idx] = g_l.detach().cpu().clone()
#                     self.gauss_info['scale_l'][idx] = scale_l.detach().cpu().clone()
#                     self.gauss_info['w_l'][idx] = w_l.detach().cpu().clone()
#                 else:
#                     # 如果不使用高斯加权，则令 w=1
#                     w_c = torch.ones(B, t_len, device=x.device, dtype=x.dtype)
#                     w_l = torch.ones(B, t_len, device=x.device, dtype=x.dtype)
#                     weights_cls.append(w_c)
#                     weights_loc.append(w_l)
#                 def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, tol=1e-6):
#                     """
#                     比较两个 tensor 的差异并打印差异的索引和数值。
                    
#                     参数：
#                         t1, t2: 待比较的两个 tensor，需形状相同
#                         tol: 判断为“不同”的最小差异阈值（默认1e-6）
#                     """
#                     if t1.shape != t2.shape:
#                         print("两个 tensor 的形状不同，无法比较。")
#                         return

#                     diff = torch.abs(t1 - t2)
#                     mask = diff > tol

#                     if mask.any():
#                         idx = torch.nonzero(mask, as_tuple=False)
#                         print("差异位置及值如下：")
#                         for i in idx:
#                             i_tuple = tuple(i.tolist())
#                             val1 = t1[i_tuple].item()
#                             val2 = t2[i_tuple].item()
#                             print(f"索引 {i_tuple}: t1={val1:.6f}, t2={val2:.6f}, 差值={abs(val1 - val2):.6f}")
#                     else:
#                         print("两个 tensor 在给定容差下完全一致。")
#                 # tmp = g_c.detach().clone()
#                 # tmp = tmp[0]
#                 # for i in range(tmp.shape[0]):
#                 #     for j in range(i+1, tmp.shape[0]):
#                 #         compare_tensors(tmp[i], tmp[2], tol=1e-6)
#                 #         exit()
#                 # 4.3 用加权后的 Q 输入到当前注意力层
#                 # print("w_c min max: ", w_c.min().item(), w_c.max().item())
#                 Q_c = feature_st_c * w_c.unsqueeze(1)  # [B, C, t_len] -> [B, C, t_len]
#                 # print("Q_c min max: ", Q_c.min().item(), Q_c.max().item())
#                 # exit()
#                 feature_st_c = layer_c(feature_st_c, Q_c, mask)  # [B, C, t_len]

#                 Q_l = feature_st_l * w_l.unsqueeze(1)  # [B, C, t_len] -> [B, C, t_len]
#                 feature_st_l = layer_l(feature_st_l, Q_l, mask)  # [B, C, t_len]

#                 # 4.4 保存每层输出，后面会做加权融合
#                 all_feats_c.append(feature_st_c)
#                 all_feats_l.append(feature_st_l)

#         # 5. 多层特征融合：先对每层做时间轴上的全局池化 -> [B, C]，拼成 [B, C*num_layers]，再让小网络预测层权重
#         # feats_c_for_weight = []
#         # feats_l_for_weight = []
#         # for idx in range(len(all_feats_c)):
#         #     tmp_c = F.adaptive_avg_pool1d(all_feats_c[idx], 1).view(B, C)  # [B, C]
#         #     tmp_l = F.adaptive_avg_pool1d(all_feats_l[idx], 1).view(B, C)
#         #     feats_c_for_weight.append(tmp_c)
#         #     feats_l_for_weight.append(tmp_l)

#         # cat_c = torch.cat(feats_c_for_weight, dim=1)  # [B, C * num_layers]
#         # cat_l = torch.cat(feats_l_for_weight, dim=1)

#         # layer_weights_c = self.layer_weight_net_c(cat_c)  # [B, num_layers]
#         # layer_weights_l = self.layer_weight_net_l(cat_l)  # [B, num_layers]
#         all_feats_c = torch.stack(all_feats_c, dim=1)  # [B, num_layers, C, t_len]
#         all_feats_l = torch.stack(all_feats_l, dim=1)  # [B, num_layers, C, t_len]
#         final_feat_c = all_feats_c.mean(1)  # 最后一层的输出作为最终特征
#         final_feat_l = all_feats_l.mean(1)  # 最后一层的输出作为最终特征
#         # for idx in range(len(all_feats_c)):
#         #     w_layer_c = layer_weights_c[:, idx].view(B, 1, 1)  # [B,1,1]
#         #     w_layer_l = layer_weights_l[:, idx].view(B, 1, 1)
#         #     final_feat_c = final_feat_c + all_feats_c[idx] * w_layer_c  # [B, C, t_len]
#         #     final_feat_l = final_feat_l + all_feats_l[idx] * w_layer_l

#         # 6. 输出
#         out_c = self.conv_out_c(final_feat_c) * mask  # [B, out_channel, t_len]
#         out_l = self.conv_out_l(final_feat_l) * mask  # [B, out_channel, t_len]

#         return out_c, out_l, div_loss * self.div_weight

    
# class STI(nn.Module):
#     def __init__(
#         self, node, in_channel, n_features, out_channel,
#         num_layers, SFI_layer, num_gauss=20, channel_masking_rate=0.3,
#         alpha=1, div_margin=0.1, div_weight=0.1, pool_len=512,
#         use_gaussian_weighting=True, use_diversity_loss=True
#     ):
#         super().__init__()
#         self.SFI_layer = SFI_layer
#         self.L = len(SFI_layer) + 1
#         self.channel_masking_rate = channel_masking_rate
#         self.dropout = nn.Dropout2d(p=channel_masking_rate)

#         self.use_gaussian_weighting = use_gaussian_weighting
#         self.use_diversity_loss = use_diversity_loss

#         # initial convs
#         self.conv_in = nn.Conv2d(in_channel, self.L, kernel_size=1)
#         self.conv_t  = nn.Conv1d(node, n_features, 1)

#         # fusion and attention layers
#         self.SFI_layers_c = nn.ModuleList([SFI(node, n_features) for _ in range(self.L-1)])
#         self.SFI_layers_l = nn.ModuleList([SFI(node, n_features) for _ in range(self.L-1)])
        
#         self.layers_c     = nn.ModuleList([AttModule(2**i, n_features, n_features, 'encoder', alpha)
#                                          for i in range(num_layers)])
#         self.layers_l     = nn.ModuleList([AttModule(2**i, n_features, n_features, 'encoder', alpha)
#                                          for i in range(num_layers)])
        
#         # cross-attention modules for each layer
#         # self.cross_attn_c = nn.ModuleList([
#         #     nn.MultiheadAttention(embed_dim=n_features, num_heads=4, batch_first=True)
#         #     for _ in range(num_layers)
#         # ])

#         # self.cross_attn_l = nn.ModuleList([
#         #     nn.MultiheadAttention(embed_dim=n_features, num_heads=4, batch_first=True)
#         #     for _ in range(num_layers)
#         # ])

#         # Gaussian mixture modeling params
#         self.num_gauss  = num_gauss
#         self.eps        = 1e-6
#         self.div_margin = div_margin
#         self.div_weight = div_weight
#         self.pool_len   = pool_len
#         eye = torch.eye(num_gauss, dtype=torch.bool)
#         self.register_buffer('div_mask', ~eye)
#         self.gauss_cls_pred = None
#         self.gauss_loc_pred = None
#         hidden = n_features * num_gauss // 2
#         self.scale_mlp_c = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(n_features * num_gauss, hidden),
#                 nn.ReLU(),
#                 nn.Linear(hidden, num_gauss),
#                 nn.Softplus()
#             ) for _ in range(num_layers)
#         ])
#         self.scale_mlp_l = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(n_features * num_gauss, hidden),
#                 nn.ReLU(),
#                 nn.Linear(hidden, num_gauss),
#                 nn.Softplus()
#             ) for _ in range(num_layers)
#         ])
#         # output convs
#         self.conv_stack_c = nn.ModuleList(
#             [nn.Conv1d(n_features, n_features, 1) for _ in range(num_layers)]
#         )
#         self.conv_stack_l = nn.ModuleList(
#             [nn.Conv1d(n_features, n_features, 1) for _ in range(num_layers)]
#         )
#         # final output convs
#         self.conv_out_c = nn.Conv1d(n_features, out_channel, 1)
#         self.conv_out_l = nn.Conv1d(n_features, out_channel, 1)

#         self.gauss_info = {
#             'mu_c': [None] * num_layers,
#             'sigma_c': [None] * num_layers,
#             'g_c': [None] * num_layers,
#             'scale_c': [None] * num_layers,
#             'w_c': [None] * num_layers,
#             'mu_l': [None] * num_layers,
#             'sigma_l': [None] * num_layers,
#             'g_l': [None] * num_layers,
#             'scale_l': [None] * num_layers,
#             'w_l': [None] * num_layers
#         }

#     def _init_gauss_predictors(self, desc_dim):
#         self.gauss_cls_pred = nn.ModuleList([
#             nn.Linear(desc_dim, 2 * self.num_gauss) for _ in range(self.L)
#         ])
#         self.gauss_loc_pred = nn.ModuleList([
#             nn.Linear(desc_dim, 2 * self.num_gauss) for _ in range(self.L)
#         ])
#         self.add_module('gauss_cls_pred', self.gauss_cls_pred)
#         self.add_module('gauss_loc_pred', self.gauss_loc_pred)

#     def forward(self, x, mask, joint_text_embedding=None):
#         # channel dropout
#         if self.channel_masking_rate > 0:
#             x = self.dropout(x)

#         # split and initial projection
#         x = self.conv_in(x)
#         feat_s, feat_t = torch.split(x, (self.L-1,1), dim=1)
#         feat_t = feat_t.squeeze(1).permute(0,2,1)
#         feature_st = self.conv_t(feat_t)

#         # prepare Gaussian predictors
#         n, C, t_len = feature_st.shape
#         if (self.use_gaussian_weighting or self.use_diversity_loss) and self.gauss_cls_pred is None:
#             desc_dim = C * min(self.pool_len, t_len)
#             self._init_gauss_predictors(desc_dim)
#             for m in self.gauss_cls_pred: m.to(x.device)
#             for m in self.gauss_loc_pred: m.to(x.device)

#         div_loss = 0.0
#         weights_cls, weights_loc = [], []
#         scale_feats_c, scale_feats_l = [], []
#         feature_st_c, feature_st_l = feature_st, feature_st
#         # iterate through layers
#         for idx, (layer_c, layer_l) in enumerate(zip(self.layers_c, self.layers_l)):
#             # SFI fusion
#             if idx in self.SFI_layer:
#                 sfi_idx = self.SFI_layer.index(idx)
#                 feature_st_c = self.SFI_layers_c[sfi_idx](feat_s[:,sfi_idx], feature_st_c, mask)
#                 feature_st_l = self.SFI_layers_c[sfi_idx](feat_s[:,sfi_idx], feature_st_l, mask)

#             # compute Gaussian weights
#             if self.use_gaussian_weighting or self.use_diversity_loss:
#                 pooled_c = F.adaptive_avg_pool1d(feature_st_c.contiguous(), self.pool_len).reshape(n, -1)
#                 # print(pooled_c.shape)
#                 # classification params
#                 params_c = self.gauss_cls_pred[idx](pooled_c)
#                 mu_c, logvar_c = params_c.chunk(2, dim=1)
#                 # mu_norm = torch.sigmoid(mu_c) * (t_len - 1)
#                 # mu_norm = torch.sigmoid(mu_c)
#                 mu_norm = torch.sigmoid(mu_c) * (t_len - 1) / self.num_gauss
#                 # print("mu_norm:", mu_norm)
#                 sigma_c = F.softplus(logvar_c) + self.eps
#                 if self.use_diversity_loss:
#                     diff = torch.relu(self.div_margin - (mu_c.unsqueeze(2) - mu_c.unsqueeze(1)).abs())
#                     div_loss += diff[:, self.div_mask].mean()
#                 mu_c = mu_norm.unsqueeze(-1)
#                 sig_c = sigma_c.unsqueeze(-1)
#                 time_idx = torch.arange(t_len, device=x.device).view(1,1,t_len).float() / self.num_gauss
#                 g_c = torch.exp(- (time_idx - mu_c)**2 / (2 * sig_c**2 + self.eps))
#                 # w_c = g_c.sum(1)  # (n, t_len)
#                 fg_c = torch.einsum('bct,bgt->btcg', feature_st_c, g_c).reshape(n*t_len, C*self.num_gauss)
#                 # def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, tol=1e-6):
#                 #     """
#                 #     比较两个 tensor 的差异并打印差异的索引和数值。
                    
#                 #     参数：
#                 #         t1, t2: 待比较的两个 tensor，需形状相同
#                 #         tol: 判断为“不同”的最小差异阈值（默认1e-6）
#                 #     """
#                 #     if t1.shape != t2.shape:
#                 #         print("两个 tensor 的形状不同，无法比较。")
#                 #         return

#                 #     diff = torch.abs(t1 - t2)
#                 #     mask = diff > tol

#                 #     if mask.any():
#                 #         idx = torch.nonzero(mask, as_tuple=False)
#                 #         print("差异位置及值如下：")
#                 #         for i in idx:
#                 #             i_tuple = tuple(i.tolist())
#                 #             val1 = t1[i_tuple].item()
#                 #             val2 = t2[i_tuple].item()
#                 #             print(f"索引 {i_tuple}: t1={val1:.6f}, t2={val2:.6f}, 差值={abs(val1 - val2):.6f}")
#                 #     else:
#                 #         print("两个 tensor 在给定容差下完全一致。")
#                 # tmp = g_c.detach().clone()
#                 # tmp = tmp[0]
#                 # for i in range(tmp.shape[0]):
#                 #     for j in range(i+1, tmp.shape[0]):
#                 #         compare_tensors(tmp[i], tmp[2], tol=1e-6)
#                 #         exit()
#                 scale_c = self.scale_mlp_c[idx](fg_c).view(n, self.num_gauss, t_len)
#                 w_c = torch.einsum('bgt,bgt->bt', g_c, scale_c)
#                 w_c = torch.clamp(w_c, min=0.1, max=1.0)  # ensure weights are in [0, 1]
#                 weights_cls.append(w_c)
#                 # localization params
#                 pooled_l = F.adaptive_avg_pool1d(feature_st_l.contiguous(), self.pool_len).reshape(n, -1)
#                 params_l = self.gauss_loc_pred[idx](pooled_l)
#                 mu_l, logvar_l = params_l.chunk(2, dim=1)
#                 sigma_l = F.softplus(logvar_l) + self.eps
#                 mu_lnorm = torch.sigmoid(mu_l) * (t_len - 1) / self.num_gauss
#                 if self.use_diversity_loss:
#                     diff2 = torch.relu(self.div_margin - (mu_l.unsqueeze(2) - mu_l.unsqueeze(1)).abs())
#                     div_loss += diff2[:, self.div_mask].mean()
                
#                 mu_l = mu_lnorm.unsqueeze(-1)
#                 sig_l = sigma_l.unsqueeze(-1)
#                 g_l = torch.exp(- (time_idx - mu_l)**2 / (2 * sig_l**2 + self.eps))
#                 # w_l = g_l.sum(1)
#                 fg_l = torch.einsum('bct,bgt->btcg', feature_st_l, g_l).reshape(n*t_len, C*self.num_gauss)
#                 scale_l = self.scale_mlp_l[idx](fg_l).view(n, self.num_gauss, t_len)
#                 w_l = torch.einsum('bgt,bgt->bt', g_l, scale_l)
#                 w_l = torch.clamp(w_l, min=0.1, max=1.0)  # ensure weights are in [0, 1]
#                 weights_loc.append(w_l)

#                 # prepare Q, K, V for cross-attention
#                 # feature_st: (n, C, t_len) -> (n, t_len, C)
#                 Q_c = feature_st_c * w_c.unsqueeze(1)
#                 # K = feature_st_c.permute(0,2,1)
#                 # V = K
#                 # # apply cross-attention
#                 # attn_out_c, _ = self.cross_attn_c[idx](Q, K, V)
#                 # # back to (n, C, t_len)
#                 # feature_st_c = attn_out_c.permute(0,2,1)

#                 Q_l = feature_st_l * w_c.unsqueeze(1)
#                 self.gauss_info['mu_c'][idx] = mu_c.detach().cpu().clone()      # detach 防止梯度累积
#                 self.gauss_info['sigma_c'][idx] = sigma_c.detach().cpu().clone()
#                 self.gauss_info['g_c'][idx] = g_c.detach().cpu().clone()
#                 self.gauss_info['scale_c'][idx] = scale_c.detach().cpu().clone()   # (B, G, T)
#                 self.gauss_info['w_c'][idx] = w_c.detach().cpu().clone()           # (B, T)

#                 self.gauss_info['mu_l'][idx] = mu_l.detach().cpu().clone()      # detach 防止梯度累积
#                 self.gauss_info['sigma_l'][idx] = sigma_l.detach().cpu().clone()
#                 self.gauss_info['g_l'][idx] = g_l.detach().cpu().clone()
#                 self.gauss_info['scale_l'][idx] = scale_l.detach().cpu().clone()   # (B, G, T)
#                 self.gauss_info['w_l'][idx] = w_l.detach().cpu().clone()           # (B, T)
#                 # print("save gauss")

#             # pass through existing layer
#             # print(torch.max(feature_st_l), torch.min(feature_st_l))
#             # print(torch.max(Q_l), torch.min(Q_l))
#             feature_st_c = layer_c(feature_st_c, Q_c, mask)
#             feature_st_l = layer_l(feature_st_l, Q_l, mask)
#             scale_feats_c.append(feature_st_c)
#             scale_feats_l.append(feature_st_l)


#         # stack last feature for both branches
#         for i in range(len(scale_feats_c)):
#             scale_feats_c[i] = self.conv_stack_c[i](scale_feats_c[i])
#             scale_feats_l[i] = self.conv_stack_l[i](scale_feats_l[i])
#         stacked_c = torch.stack(scale_feats_c,1)
#         stacked_l = torch.stack(scale_feats_l,1)
        
#         final_feat_c = stacked_c.sum(dim=1)  # (n, C, t_len)
#         final_feat_l = stacked_l.sum(dim=1)  # (n, C, t_len)
#         # if self.use_gaussian_weighting:
#         #     # aggregate with last computed weights (optional: choose mean if varying)
#         #     w_c_last = weights_cls[-1].unsqueeze(1)
#         #     w_l_last = weights_loc[-1].unsqueeze(1)
#         #     feat_c = final_feat_c * w_c_last
#         #     feat_l = final_feat_l * w_l_last
#         # else:
#         #     feat_c = final_feat_c
#         #     feat_l = final_feat_l

#         # temporal pooling
#         # pooled_c = F.adaptive_avg_pool1d(final_feat_c, self.pool_len)
#         # pooled_l = F.adaptive_avg_pool1d(final_feat_l, self.pool_len)
#         # flat_c = pooled_c.view(n, -1)
#         # flat_l = pooled_l.view(n, -1)

#         out_c = self.conv_out_c(final_feat_c) * mask
#         out_l = self.conv_out_l(final_feat_l) * mask
#         return out_c, out_l, div_loss * 0.1


import torch
import torch.nn as nn
import torch.nn.functional as F

class STI(nn.Module):
    def __init__(
        self, node, in_channel, n_features, out_channel,
        num_layers, SFI_layer,
        M=2,                    # 每帧要预测 M 个高斯
        channel_masking_rate=0.3,
        alpha=1, div_margin=0.1, div_weight=0.1,
        pool_len=64,            # 将原始 t_len 池化到 64 帧
        use_gaussian_weighting=True, use_diversity_loss=True
    ):
        super().__init__()
        self.SFI_layer = SFI_layer
        self.L = len(SFI_layer) + 1
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate)

        self.use_gaussian_weighting = use_gaussian_weighting
        self.use_diversity_loss = use_diversity_loss

        # 初始化卷积
        self.conv_in = nn.Conv2d(in_channel, self.L, kernel_size=1)
        self.conv_t  = nn.Conv1d(node, n_features, 1)

        # SFI 融合层，与原来一致
        self.SFI_layers_c = nn.ModuleList([SFI(node, n_features) for _ in range(self.L-1)])
        self.SFI_layers_l = nn.ModuleList([SFI(node, n_features) for _ in range(self.L-1)])

        # 多层 Attention 模块，与原来一致
        self.layers_c = nn.ModuleList([AttModule(2**i, n_features, n_features, 'encoder', alpha)
                                       for i in range(num_layers)])
        self.layers_l = nn.ModuleList([AttModule(2**i, n_features, n_features, 'encoder', alpha)
                                       for i in range(num_layers)])

        # --------- 调整后的高斯混合参数部分 ---------
        # M 表示“每帧预测的高斯数量”，pool_len 表示池化后帧数（如 64）
        self.M = M
        self.pool_len = pool_len      # 先把原始 t_len 池化到 pool_len 帧
        self.G = pool_len * M         # 总共要预测的高斯数 G = 64 × M
        self.eps = 1e-6
        self.div_margin = div_margin
        self.div_weight = div_weight

        # 为 diversity loss 做掩码：在所有 G 个高斯之间计算 pairwise 时，用于去掉对角线
        eye = torch.eye(self.G, dtype=torch.bool)
        self.register_buffer('div_mask', ~eye)

        # 在 forward 里第一次需要时再初始化下面这两个 ModuleList
        # gauss_cls_pred: 用于“预测 μ_raw”  -> 对应每层的 [B, C, pool_len] -> [B, 2*M, pool_len]
        # gauss_loc_pred: 用于“预测 logσ_raw”
        self.gauss_cls_pred = None
        self.gauss_loc_pred = None

        # 用来保存每层的中间高斯信息（仅作可视化或 debug 用）
        self.gauss_info = {
            'mu_c': [None] * num_layers,
            'sigma_c': [None] * num_layers,
            'g_c': [None] * num_layers,
            'scale_c': [None] * num_layers,
            'w_c': [None] * num_layers,
            'mu_l': [None] * num_layers,
            'sigma_l': [None] * num_layers,
            'g_l': [None] * num_layers,
            'scale_l': [None] * num_layers,
            'w_l': [None] * num_layers
        }

        # “每个高斯对应的 scale” 预测器：输入维度会是 (n_features × G)，输出 G 个 scale
        hidden = n_features * self.G // 2
        self.scale_mlp_c = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features * self.G, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.G),
                nn.Softplus()
            ) for _ in range(num_layers)
        ])
        self.scale_mlp_l = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_features * self.G, hidden),
                nn.ReLU(),
                nn.Linear(hidden, self.G),
                nn.Softplus()
            ) for _ in range(num_layers)
        ])

        # ===== 多层特征融合时不再用 conv1d 累加，而是网络预测权重 =====
        self.layer_weight_net_c = nn.Sequential(
            nn.Linear(n_features * num_layers, n_features),
            nn.ReLU(),
            nn.Linear(n_features, num_layers),
            nn.Softmax(dim=1)
        )
        self.layer_weight_net_l = nn.Sequential(
            nn.Linear(n_features * num_layers, n_features),
            nn.ReLU(),
            nn.Linear(n_features, num_layers),
            nn.Softmax(dim=1)
        )

        # 最后输出分支
        self.conv_out_c = nn.Conv1d(n_features, out_channel, 1)
        self.conv_out_l = nn.Conv1d(n_features, out_channel, 1)

        self.gauss_cls_pred = nn.ModuleList([
            nn.Linear(64, 2 * self.M)  # 逐帧预测 2*M 个数
            for _ in range(self.L)
        ])
        self.gauss_loc_pred = nn.ModuleList([
            nn.Linear(64, 2 * self.M)
            for _ in range(self.L)
        ])
    def _init_gauss_predictors(self, feat_dim):
        """
        初始化两个 ModuleList，按层数分别为 gauss_cls_pred（C->2M）和 gauss_loc_pred（C->2M）。
        这里用 Conv1d 而不是 Linear，因为要对 pool 后的 [B, C, pool_len] 进行“逐帧”预测：
          - 输入：pool_feat_c->[B, C, pool_len]
          - gauss_cls_pred[idx](pool_feat_c) -> [B, 2*M, pool_len]
          - split 出 mu_raw_unscaled 与 logvar_raw_unscaled，均为 [B, M, pool_len]
        """
        self.gauss_cls_pred = nn.ModuleList([
            nn.Linear(feat_dim, 2 * self.M)  # 逐帧预测 2*M 个数
            for _ in range(self.L)
        ])
        self.gauss_loc_pred = nn.ModuleList([
            nn.Linear(feat_dim, 2 * self.M)
            for _ in range(self.L)
        ])
        self.add_module('gauss_cls_pred', self.gauss_cls_pred)
        self.add_module('gauss_loc_pred', self.gauss_loc_pred)

    def forward(self, x, mask, joint_text_embedding=None):
        """
        x: [B, in_channel, node, t_len]
        mask: [B, 1, t_len]
        """
        # 1. channel dropout
        if self.channel_masking_rate > 0:
            x = self.dropout(x)

        # 2. split feat_s / feat_t，并做最初映射
        x = self.conv_in(x)                  # [B, L, node, t_len]
        feat_s, feat_t = torch.split(x, (self.L-1, 1), dim=1)
        feat_t = feat_t.squeeze(1).permute(0, 2, 1)  # [B, node, t_len]
        feature_st = self.conv_t(feat_t)            # [B, C, t_len]
        B, C, t_len = feature_st.shape

        # 3. 第一次进入时，如果需要高斯加权或多样性损失，提前初始化预测器
        # if (self.use_gaussian_weighting or self.use_diversity_loss) and (self.gauss_cls_pred is None):
        #     # pool 到 64 帧：feat_dim = C（因为我们用 Conv1d 逐帧预测，所以只需 C 作为输入通道数）
        #     feat_dim = C
        #     # self._init_gauss_predictors(feat_dim)
        #     for m in self.gauss_cls_pred: m.to(x.device)
        #     for m in self.gauss_loc_pred: m.to(x.device)

        div_loss = 0.0
        weights_cls, weights_loc = [], []
        all_feats_c, all_feats_l = [], []

        feature_st_c, feature_st_l = feature_st, feature_st

        for idx, (layer_c, layer_l) in enumerate(zip(self.layers_c, self.layers_l)):
            # 4.1 如果在 SFI 层，则先做跨模态融合
            if idx in self.SFI_layer:
                sfi_idx = self.SFI_layer.index(idx)
                feature_st_c = self.SFI_layers_c[sfi_idx](feat_s[:, sfi_idx], feature_st_c, mask)
                feature_st_l = self.SFI_layers_l[sfi_idx](feat_s[:, sfi_idx], feature_st_l, mask)

            # 4.2 计算第 idx 层对应的 64 帧 × M 高斯
            if self.use_gaussian_weighting or self.use_diversity_loss:
                # 4.2.1 先把 feature_st_c 池化到 [B, C, pool_len=64]
                
                pooled_c = F.adaptive_avg_pool1d(feature_st_c.contiguous(), self.pool_len)  # [B, 64, C]
                
                # 用 Conv1d 逐帧预测 2*M 个参数
                # params_c_raw: [B, 2*M, 64]
                params_c_raw = self.gauss_cls_pred[idx](pooled_c.permute(0,2,1)).permute(0,2,1)  # [B, 2*M, 64]
                # print(idx, "params_c_raw.shape:", params_c_raw.shape)  # 应该是 [B, 2*M, pool_len]

                # 拆成 mu_raw_unscaled 与 logvar_raw_unscaled（都形状 [B, M, 64]）
                mu_c_raw_unscaled, logvar_c_raw_unscaled = params_c_raw.chunk(2, dim=1)  # 各 [B, M, 64]

                # 对 logvar 做 softplus 确保正数
                sigma_c_raw_unscaled = F.softplus(logvar_c_raw_unscaled) + self.eps  # [B, M, 64]
                # 对 mu_raw 做 sigmoid，把值归一到 (0,1) 区间内
                mu_c_unit = torch.sigmoid(mu_c_raw_unscaled)  # [B, M, 64]

                # 4.2.2 多样性损失：对这 64×M 个 μ_raw_unscaled 进行 pairwise 约束
                if self.use_diversity_loss:
                    mu_flat = mu_c_raw_unscaled.view(B, self.G)  # [B, 64*M]
                    diff = torch.relu(self.div_margin - (mu_flat.unsqueeze(2) - mu_flat.unsqueeze(1)).abs())
                    div_loss += diff[:, self.div_mask].mean()

                # 4.2.3 计算全局坐标下的 mu_abs 与 sigma_abs
                # pool_idxs 必须是 [1, 64, 1]，才能和 mu_c_unit.permute(0,2,1) ([B,64,M]) 广播对齐
                device = x.device
                pool_idxs = torch.arange(self.pool_len, device=device).view(1, self.pool_len, 1)  # [1, 64, 1]
                seg_len = t_len / float(self.pool_len)  # 每段的实际帧数

                # permute mu_c_unit 从 [B, M, 64] -> [B, 64, M]
                mu_c_unit = mu_c_unit.permute(0, 2, 1)       # [B, 64, M]
                sigma_c_unit = sigma_c_raw_unscaled.permute(0, 2, 1)  # [B, 64, M]

                # 先平移（broadcast 到 [B,64,M] 加上 pool_idxs：[1,64,1]），再放缩
                mu_c_abs = (pool_idxs + mu_c_unit) * seg_len   # [B, 64, M]
                sigma_c_abs = sigma_c_unit * seg_len           # [B, 64, M]

                # 将 [B,64,M] reshape/flatten 为 [B, G, 1]（G = 64*M），方便后续计算
                mu_c_abs = mu_c_abs.reshape(B, self.G, 1)      # [B, G, 1]
                sigma_c_abs = sigma_c_abs.reshape(B, self.G, 1)# [B, G, 1]

                # time_idx: [1, 1, t_len]
                time_idx = torch.arange(t_len, device=device).view(1, 1, t_len).float()

                # 4.2.4 计算高斯权重 g_c: [B, G, t_len]
                g_c = torch.exp(- (time_idx - mu_c_abs)**2 / (2 * sigma_c_abs**2 + self.eps))  # [B, G, t_len]

                # 4.2.5 计算与 feature_st_c 的加权特征并预测 scale_c
                # feature_st_c: [B, C, t_len], g_c: [B, G, t_len]
                # fg_c_raw = torch.einsum('bct,bgt->btcg', feature_st_c, g_c).reshape(B * t_len, C * self.G)
                # scale_c = self.scale_mlp_c[idx](fg_c_raw).view(B, self.G, t_len)  # [B, G, t_len]
                scale_c = torch.ones((B, self.G, t_len), device=x.device, dtype=x.dtype)  # 初始化为全 1

                # 4.2.6 w_c: [B, t_len] = sum over G of (g_c * scale_c)
                w_c = torch.einsum('bgt,bgt->bt', g_c, scale_c)  # [B, t_len]
                w_c = torch.clamp(w_c, min=0.1, max=1.0)         # 保证在 [0.1, 1.0] 之间
                weights_cls.append(w_c)

                # 4.2.7 对 feature_st_l 做同样的操作
                pooled_l = F.adaptive_avg_pool1d(feature_st_l.contiguous(), self.pool_len)  # [B, C, 64]
                params_l_raw = self.gauss_loc_pred[idx](pooled_l.permute(0,2,1)).permute(0,2,1)  # [B, 2*M, 64]
                mu_l_raw_unscaled, logvar_l_raw_unscaled = params_l_raw.chunk(2, dim=1)  # 各 [B, M, 64]
                sigma_l_raw_unscaled = F.softplus(logvar_l_raw_unscaled) + self.eps       # [B, M, 64]
                mu_l_unit = torch.sigmoid(mu_l_raw_unscaled)                             # [B, M, 64]

                if self.use_diversity_loss:
                    mu_l_flat = mu_l_raw_unscaled.view(B, self.G)  # [B, G]
                    diff2 = torch.relu(self.div_margin - (mu_l_flat.unsqueeze(2) - mu_l_flat.unsqueeze(1)).abs())
                    div_loss += diff2[:, self.div_mask].mean()

                mu_l_unit = mu_l_unit.permute(0, 2, 1)        # [B, 64, M]
                sigma_l_unit = sigma_l_raw_unscaled.permute(0, 2, 1)  # [B,64,M]

                mu_l_abs = (pool_idxs + mu_l_unit) * seg_len  # [B, 64, M]
                sigma_l_abs = sigma_l_unit * seg_len          # [B, 64, M]

                mu_l_abs = mu_l_abs.reshape(B, self.G, 1)     # [B, G, 1]
                sigma_l_abs = sigma_l_abs.reshape(B, self.G, 1)  # [B, G, 1]

                g_l = torch.exp(- (time_idx - mu_l_abs)**2 / (2 * sigma_l_abs**2 + self.eps))  # [B, G, t_len]

                # fg_l_raw = torch.einsum('bct,bgt->btcg', feature_st_l, g_l).reshape(B * t_len, C * self.G)
                # scale_l = self.scale_mlp_l[idx](fg_l_raw).view(B, self.G, t_len)  # [B, G, t_len]
                scale_l = torch.ones((B, self.G, t_len), device=x.device, dtype=x.dtype)  # 初始化为全 1
                w_l = torch.einsum('bgt,bgt->bt', g_l, scale_l)  # [B, t_len]
                w_l = torch.clamp(w_l, min=0.1, max=1.0)        # [B, t_len]
                weights_loc.append(w_l)

                # 4.2.8 存储调试/可视化用的高斯信息
                self.gauss_info['mu_c'][idx] = mu_c_abs.detach().cpu().clone()
                self.gauss_info['sigma_c'][idx] = sigma_c_abs.detach().cpu().clone()
                self.gauss_info['g_c'][idx] = g_c.detach().cpu().clone()
                self.gauss_info['scale_c'][idx] = scale_c.detach().cpu().clone()
                self.gauss_info['w_c'][idx] = w_c.detach().cpu().clone()

                self.gauss_info['mu_l'][idx] = mu_l_abs.detach().cpu().clone()
                self.gauss_info['sigma_l'][idx] = sigma_l_abs.detach().cpu().clone()
                self.gauss_info['g_l'][idx] = g_l.detach().cpu().clone()
                self.gauss_info['scale_l'][idx] = scale_l.detach().cpu().clone()
                self.gauss_info['w_l'][idx] = w_l.detach().cpu().clone()
            else:
                # 如果不使用高斯加权，则令 w=1
                w_c = torch.ones(B, t_len, device=x.device, dtype=x.dtype)
                w_l = torch.ones(B, t_len, device=x.device, dtype=x.dtype)
                weights_cls.append(w_c)
                weights_loc.append(w_l)

            # 4.3 用加权后的 Q 输入到当前注意力层
            Q_c = feature_st_c * w_c.unsqueeze(1)  
            feature_st_c = layer_c(feature_st_c, Q_c, mask)  # [B, C, t_len]

            Q_l = feature_st_l * w_l.unsqueeze(1)
            feature_st_l = layer_l(feature_st_l, Q_l, mask)  # [B, C, t_len]

            # 4.4 保存每层输出，后面会做加权融合
            all_feats_c.append(feature_st_c)
            all_feats_l.append(feature_st_l)

        # 5. 多层特征融合：先对每层做时间轴上的全局池化 -> [B, C]，拼成 [B, C*num_layers]，再让小网络预测层权重
        # feats_c_for_weight = []
        # feats_l_for_weight = []
        # for idx in range(len(all_feats_c)):
        #     tmp_c = F.adaptive_avg_pool1d(all_feats_c[idx], 1).view(B, C)  # [B, C]
        #     tmp_l = F.adaptive_avg_pool1d(all_feats_l[idx], 1).view(B, C)
        #     feats_c_for_weight.append(tmp_c)
        #     feats_l_for_weight.append(tmp_l)

        # cat_c = torch.cat(feats_c_for_weight, dim=1)  # [B, C * num_layers]
        # cat_l = torch.cat(feats_l_for_weight, dim=1)

        # layer_weights_c = self.layer_weight_net_c(cat_c)  # [B, num_layers]
        # layer_weights_l = self.layer_weight_net_l(cat_l)  # [B, num_layers]

        # final_feat_c = 0
        # final_feat_l = 0
        all_feats_c = torch.stack(all_feats_c, dim=1)  # [B, num_layers, C, t_len]
        all_feats_l = torch.stack(all_feats_l, dim=1)  # [B, num_layers, C, t_len]
        # for idx in range(len(all_feats_c)):
        #     w_layer_c = layer_weights_c[:, idx].view(B, 1, 1)  # [B,1,1]
        #     w_layer_l = layer_weights_l[:, idx].view(B, 1, 1)
        #     final_feat_c = final_feat_c + all_feats_c[idx] * w_layer_c  # [B, C, t_len]
        #     final_feat_l = final_feat_l + all_feats_l[idx] * w_layer_l
        final_feat_c = all_feats_c.sum(1)
        final_feat_l = all_feats_l.sum(1)

        # 6. 输出
        out_c = self.conv_out_c(final_feat_c) * mask  # [B, out_channel, t_len]
        out_l = self.conv_out_l(final_feat_l) * mask  # [B, out_channel, t_len]

        return out_c, out_l, div_loss * self.div_weight


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class STI(nn.Module):
#     def __init__(
#         self, node, in_channel, n_features, out_channel,
#         num_layers, SFI_layer,
#         M=1,                    # 每帧要预测 M 个高斯
#         channel_masking_rate=0.3,
#         alpha=1, div_margin=0.1, div_weight=0.1,
#         pool_len=64,            # 将原始 t_len 池化到 64 帧，对应新的 0–63 轴
#         use_gaussian_weighting=True, use_diversity_loss=True
#     ):
#         super().__init__()
#         self.SFI_layer = SFI_layer
#         self.L = len(SFI_layer) + 1
#         self.channel_masking_rate = channel_masking_rate
#         self.dropout = nn.Dropout2d(p=channel_masking_rate)

#         self.use_gaussian_weighting = use_gaussian_weighting
#         self.use_diversity_loss = use_diversity_loss

#         # 初始卷积：把 in_channel→L；再把 node→n_features
#         self.conv_in = nn.Conv2d(in_channel, self.L, kernel_size=1)
#         self.conv_t  = nn.Conv1d(node, n_features, 1)

#         # SFI 融合层，与原来一致
#         self.SFI_layers_c = nn.ModuleList([SFI(node, n_features) for _ in range(self.L-1)])
#         self.SFI_layers_l = nn.ModuleList([SFI(node, n_features) for _ in range(self.L-1)])

#         # 多层 Attention 模块，与原来一致
#         self.layers_c = nn.ModuleList([
#             AttModule(2**i, n_features, n_features, 'encoder', alpha)
#             for i in range(num_layers)
#         ])
#         self.layers_l = nn.ModuleList([
#             AttModule(2**i, n_features, n_features, 'encoder', alpha)
#             for i in range(num_layers)
#         ])

#         # --------- 调整后的高斯混合参数部分 ---------
#         # M 表示“每帧预测的高斯数量”，pool_len 表示新的高斯参考长度（0–pool_len-1）
#         self.M = M
#         self.pool_len = pool_len      # pool_len=64，后面把时间轴映射到 [0,63]
#         self.G = pool_len * M         # 总共要预测的高斯数 G = 64 × M
#         self.eps = 1e-6
#         self.div_margin = div_margin
#         self.div_weight = div_weight
#         self.alpha = alpha            # 用于控制高斯陡峭程度

#         # 为 diversity loss 做掩码：在所有 G 个高斯之间计算 pairwise 时，用于去掉对角线
#         eye = torch.eye(self.G, dtype=torch.bool)
#         self.register_buffer('div_mask', ~eye)

#         # 延后在 forward 初始化下面这两个列表
#         # gauss_cls_pred: 用于“预测 μ_raw”（输出 [B, 2*M, pool_len]）
#         # gauss_loc_pred: 用于“预测 σ_raw”（输出 [B, 2*M, pool_len]）
#         self.gauss_cls_pred = None
#         self.gauss_loc_pred = None

#         # 保存每层的中间高斯信息（可视化/调试用）
#         self.gauss_info = {
#             'mu_c': [None] * num_layers,
#             'sigma_c': [None] * num_layers,
#             'g_c': [None] * num_layers,
#             'w_c': [None] * num_layers,
#             'mu_l': [None] * num_layers,
#             'sigma_l': [None] * num_layers,
#             'g_l': [None] * num_layers,
#             'w_l': [None] * num_layers
#         }

#         # “每个高斯对应的 scale” 预测器：输入维度是 (n_features × G)，输出 G 个 scale
#         hidden = n_features * self.G // 2
#         self.scale_mlp_c = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(n_features * self.G, hidden),
#                 nn.ReLU(),
#                 nn.Linear(hidden, self.G),
#                 nn.Softplus()
#             ) for _ in range(num_layers)
#         ])
#         self.scale_mlp_l = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(n_features * self.G, hidden),
#                 nn.ReLU(),
#                 nn.Linear(hidden, self.G),
#                 nn.Softplus()
#             ) for _ in range(num_layers)
#         ])

#         # ===== 多层特征融合时不再用 conv1d 累加，而是网络预测层权重 =====
#         self.layer_weight_net_c = nn.Sequential(
#             nn.Linear(n_features * num_layers, n_features),
#             nn.ReLU(),
#             nn.Linear(n_features, num_layers),
#             nn.Softmax(dim=1)
#         )
#         self.layer_weight_net_l = nn.Sequential(
#             nn.Linear(n_features * num_layers, n_features),
#             nn.ReLU(),
#             nn.Linear(n_features, num_layers),
#             nn.Softmax(dim=1)
#         )

#         # 最后输出分支
#         self.conv_out_c = nn.Conv1d(n_features, out_channel, 1)
#         self.conv_out_l = nn.Conv1d(n_features, out_channel, 1)

#     def _init_gauss_predictors(self, feat_dim):
#         """
#         初始化两个 ModuleList，按层数分别为：
#           - gauss_cls_pred[idx]: Conv1d(feat_dim → 2*M)，对 [B,C,pool_len] 中的每个“池化索引”预测 2*M 个参数（μ_raw 与 σ_raw）
#           - gauss_loc_pred[idx]: Conv1d(feat_dim → 2*M)，同样对 [B,C,pool_len] 预测 σ_raw
#         """
#         self.gauss_cls_pred = nn.ModuleList([
#             nn.Conv1d(feat_dim, 2 * self.M, kernel_size=1)  # 负责输出 μ_raw 和 σ_raw
#             for _ in range(self.L)
#         ])
#         self.gauss_loc_pred = nn.ModuleList([
#             nn.Conv1d(feat_dim, 2 * self.M, kernel_size=1)
#             for _ in range(self.L)
#         ])
#         self.add_module('gauss_cls_pred', self.gauss_cls_pred)
#         self.add_module('gauss_loc_pred', self.gauss_loc_pred)

#     def forward(self, x, mask, joint_text_embedding=None):
#         """
#         x: [B, in_channel, node, t_len]
#         mask: [B, 1, t_len]
#         """
#         # 1. channel dropout
#         if self.channel_masking_rate > 0:
#             x = self.dropout(x)

#         # 2. split feat_s / feat_t，并做最初映射
#         x = self.conv_in(x)                  # [B, L, node, t_len]
#         feat_s, feat_t = torch.split(x, (self.L-1, 1), dim=1)
#         feat_t = feat_t.squeeze(1).permute(0, 2, 1)  # [B, node, t_len]
#         feature_st = self.conv_t(feat_t)            # [B, C, t_len]
#         B, C, t_len = feature_st.shape

#         # 3. 第一次进入时，如果需要高斯加权或多样性损失，就初始化预测器
#         if (self.use_gaussian_weighting or self.use_diversity_loss) and (self.gauss_cls_pred is None):
#             feat_dim = C  # 因为我们要对 [B, C, pool_len] 做 Conv1d → [B, 2*M, pool_len]
#             self._init_gauss_predictors(feat_dim)
#             for m in self.gauss_cls_pred: m.to(x.device)
#             for m in self.gauss_loc_pred: m.to(x.device)

#         div_loss = 0.0
#         weights_cls, weights_loc = [], []
#         all_feats_c, all_feats_l = [], []

#         feature_st_c, feature_st_l = feature_st, feature_st

#         # —— 4. 遍历每一层：SFI + “高斯加权” + Attention —— #
#         for idx, (layer_c, layer_l) in enumerate(zip(self.layers_c, self.layers_l)):
#             # 4.1 如果在 SFI_layer 中，则先做跨模态融合
#             if idx in self.SFI_layer:
#                 sfi_idx = self.SFI_layer.index(idx)
#                 feature_st_c = self.SFI_layers_c[sfi_idx](feat_s[:, sfi_idx], feature_st_c, mask)
#                 feature_st_l = self.SFI_layers_l[sfi_idx](feat_s[:, sfi_idx], feature_st_l, mask)

#             # 4.2 计算第 idx 层对应的 64 帧 × M 个高斯（单次完成，不再拆块）
#             if self.use_gaussian_weighting or self.use_diversity_loss:
#                 # —— 4.2.1 先把 feature_st_c 池化到 [B, C, pool_len=64] —— #
#                 pooled_c = F.adaptive_avg_pool1d(feature_st_c.contiguous(), self.pool_len)  # [B, C, 64]

#                 # —— 4.2.2 逐帧预测 μ_raw_unscaled, logσ_raw_unscaled，均为 [B, M, 64] —— #
#                 params_c_raw = self.gauss_cls_pred[idx](pooled_c)      # [B, 2*M, 64]
#                 mu_c_raw, sigma_c_raw = params_c_raw.chunk(2, dim=1)   # 各 [B, M, 64]
#                 # 先对 μ_raw 做 sigmoid 归一化到 [0,1]，再乘 (pool_len-1)=63，将其映射到新时序轴 [0,63]
#                 mu_c_unit = torch.sigmoid(mu_c_raw)                     # [B, M, 64]
#                 mu_c_abs  = mu_c_unit * (self.pool_len - 1)             # [B, M, 64]

#                 # 对 σ_raw 也做 sigmoid 再 clamp，使其在 [0.1, 1.0] 之间
#                 sigma_c_unit = torch.sigmoid(sigma_c_raw)               # [B, M, 64]
#                 sigma_c_abs  = torch.clamp(sigma_c_unit, min=0.1, max=1.0)  # [B, M, 64]
#                 # 解释：这里 “每个高斯主要负责区间宽度 ~1”，
#                 # 我们让 σ ∈ [0.1, 1.0]，如果想固定为 1.0，可以直接写 sigma_c_abs = 1.0

#                 # —— 4.2.3 Diversity Loss —— #
#                 if self.use_diversity_loss:
#                     mu_flat = mu_c_raw.view(B, self.G)  # [B, 64*M]
#                     diff = torch.relu(self.div_margin - (mu_flat.unsqueeze(2) - mu_flat.unsqueeze(1)).abs())
#                     div_loss += diff[:, self.div_mask].mean()

#                 # —— 4.2.4 在新的 0–63 轴上生成高斯权重 g_c，形状 [B, G, t_len] —— #
#                 # t_map: 把原始 t_len 映射到 [0, pool_len-1] 上
#                 # time_idx 原本是 [0,1,2,...,t_len-1]，先归一到 [0,1]，再乘 (pool_len-1)
#                 device = x.device
#                 time_idx = torch.linspace(0, 1, t_len, device=device).view(1, 1, t_len)  # [1,1,t_len]
#                 time_idx = time_idx * (self.pool_len - 1)  # 归一化到 [0,63] 轴，形状仍 [1,1,t_len]

#                 # 把 mu_c_abs 从 [B, M, 64] → 重排到 [B, 64*M] 再 unsqueeze[-1] 变成 [B, G, 1]
#                 mu_c_abs = mu_c_abs.permute(0, 2, 1).reshape(B, self.G, 1)     # [B, G, 1]
#                 sigma_c_abs = sigma_c_abs.permute(0, 2, 1).reshape(B, self.G, 1)  # [B, G, 1]

#                 # 计算高斯权重：g_c[b, g, t] = exp(-α * (t_map[t] - μ_c_abs[b,g])^2 / (2 * σ_c_abs[b,g]^2))
#                 g_c = torch.exp(- self.alpha * (time_idx - mu_c_abs)**2 / (2 * sigma_c_abs**2 + self.eps))  # [B, G, t_len]

#                 # —— 4.2.5 与 feature_st_c 做加权并预测 scale_c —— #
#                 # feature_st_c: [B, C, t_len], g_c: [B, G, t_len]
#                 fg_c = torch.einsum('bct,bgt->btcg', feature_st_c, g_c)  # [B, t_len, C, G]
#                 fg_c = fg_c.reshape(B * t_len, C * self.G)                # [B*t_len, C*G]
#                 scale_c = self.scale_mlp_c[idx](fg_c).view(B, self.G, t_len)  # [B, G, t_len]

#                 # 4.2.6 最终 w_c = sum_over_G (g_c * scale_c)，再 clamp
#                 w_c = torch.einsum('bgt,bgt->bt', g_c, scale_c)  # [B, t_len]
#                 w_c = torch.clamp(w_c, min=0.1, max=1.0)         # [B, t_len]
#                 weights_cls.append(w_c)

#                 # —— 对 feature_st_l 做同样修改 —— #
#                 pooled_l = F.adaptive_avg_pool1d(feature_st_l.contiguous(), self.pool_len)  # [B, C, 64]
#                 params_l_raw = self.gauss_loc_pred[idx](pooled_l)  # [B, 2*M, 64]
#                 mu_l_raw, sigma_l_raw = params_l_raw.chunk(2, dim=1)  # 各 [B, M, 64]

#                 mu_l_unit = torch.sigmoid(mu_l_raw)                     # [B, M, 64]
#                 mu_l_abs  = mu_l_unit * (self.pool_len - 1)             # [B, M, 64]
#                 sigma_l_unit = torch.sigmoid(sigma_l_raw)               # [B, M, 64]
#                 sigma_l_abs  = torch.clamp(sigma_l_unit, min=0.1, max=1.0)  # [B, M, 64]

#                 if self.use_diversity_loss:
#                     mu_l_flat = mu_l_raw.view(B, self.G)  # [B, 64*M]
#                     diff2 = torch.relu(self.div_margin - (mu_l_flat.unsqueeze(2) - mu_l_flat.unsqueeze(1)).abs())
#                     div_loss += diff2[:, self.div_mask].mean()

#                 mu_l_abs = mu_l_abs.permute(0, 2, 1).reshape(B, self.G, 1)    # [B, G, 1]
#                 sigma_l_abs = sigma_l_abs.permute(0, 2, 1).reshape(B, self.G, 1)  # [B, G, 1]

#                 g_l = torch.exp(- self.alpha * (time_idx - mu_l_abs)**2 / (2 * sigma_l_abs**2 + self.eps))  # [B, G, t_len]
#                 fg_l = torch.einsum('bct,bgt->btcg', feature_st_l, g_l).reshape(B * t_len, C * self.G)
#                 scale_l = self.scale_mlp_l[idx](fg_l).view(B, self.G, t_len)  # [B, G, t_len]
#                 w_l = torch.einsum('bgt,bgt->bt', g_l, scale_l)  # [B, t_len]
#                 w_l = torch.clamp(w_l, min=0.1, max=1.0)
#                 weights_loc.append(w_l)

#                 # —— 存储可视化/调试信息（可选，只保留最后一层） —— #
#                 self.gauss_info['mu_c'][idx] = mu_c_abs.detach().cpu().clone()
#                 self.gauss_info['sigma_c'][idx] = sigma_c_abs.detach().cpu().clone()
#                 self.gauss_info['g_c'][idx] = g_c.detach().cpu().clone()
#                 self.gauss_info['w_c'][idx] = w_c.detach().cpu().clone()
#                 self.gauss_info['mu_l'][idx] = mu_l_abs.detach().cpu().clone()
#                 self.gauss_info['sigma_l'][idx] = sigma_l_abs.detach().cpu().clone()
#                 self.gauss_info['g_l'][idx] = g_l.detach().cpu().clone()
#                 self.gauss_info['w_l'][idx] = w_l.detach().cpu().clone()

#             else:
#                 # 如果不使用高斯加权，则 w=1
#                 w_c = torch.ones((B, t_len), device=x.device, dtype=x.dtype)
#                 w_l = torch.ones((B, t_len), device=x.device, dtype=x.dtype)
#                 weights_cls.append(w_c)
#                 weights_loc.append(w_l)

#             # 4.3 用 w_c、w_l 加权 Q 然后过 Attention
#             Q_c = feature_st_c * w_c.unsqueeze(1)  
#             feature_st_c = layer_c(Q_c, None, mask)  # [B, C, t_len]

#             Q_l = feature_st_l * w_l.unsqueeze(1)
#             feature_st_l = layer_l(Q_l, None, mask)  # [B, C, t_len]

#             # 4.4 保存每层输出
#             all_feats_c.append(feature_st_c)
#             all_feats_l.append(feature_st_l)

#         # —— 5. 多层特征融合：最终只取最后一层作为输出（如您所示） —— #
#         final_feat_c = all_feats_c[-1]  # [B, C, t_len]
#         final_feat_l = all_feats_l[-1]  # [B, C, t_len]

#         # 6. 输出
#         out_c = self.conv_out_c(final_feat_c) * mask  # [B, out_channel, t_len]
#         out_l = self.conv_out_l(final_feat_l) * mask  # [B, out_channel, t_len]

#         return out_c, out_l, div_loss * self.div_weight

       
class Decoder(nn.Module):
    def __init__(self, in_channel, n_features, out_channel, num_layers, alpha=1):
        super().__init__()
        
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'decoder', alpha) for i in 
             range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        out = self.conv_out(feature)
        
        return out, feature

    
class Model(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_refine_layers: int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        SFI_layer: Optional[int] = None,
        dataset: str = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones(2) * np.log(1 / 0.07))  # 2.6593

        self.in_channel = in_channel
        node = 19 if dataset == "LARA" else 25

        self.SP = MultiScale_GraphConv(13, in_channel, n_features, dataset) #MS-G3D
        self.STI = STI(node, n_features, n_features, \
                       n_features, n_layers, SFI_layer,\
                        use_diversity_loss=False, 
                        use_gaussian_weighting=True, 
                        )
        self.joint_att = Spatial_AttLayer(n_features, n_features, 32, 64, 4)
 
        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)
        self.conv_feature = nn.Conv1d(n_features,512,1)
        self.conv_feature_split = nn.Conv1d(n_features, 512, 1)

        # action segmentation branch
        asb = [
            copy.deepcopy(Decoder(n_classes, n_features, n_classes, n_refine_layers, alpha=exponential_descrease(s))) for s in range(n_stages_asb - 1)
        ]
        conv_asb_feature = [nn.Conv1d(n_features,512,1) for s in range(n_stages_asb - 1)]
        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_refine_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)
        self.conv_asb_feature = nn.ModuleList(conv_asb_feature)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()
        self.ff_embedding = nn.Linear(512, 512)
        self.div_loss = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor, joint_text_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        joint_text_embedding = self.ff_embedding(joint_text_embedding)
        joint_text_embedding = joint_text_embedding.permute(1, 0).unsqueeze(0).unsqueeze(2)
        # print(x.shape) #  torch.Size([3, 12, 19, 6000])

        x = self.SP(x) * mask.unsqueeze(3) #MS-G3D （n,c,t,v）
        # print(x.shape) #  torch.Size([3, 64, 6000, 19])
        # exit()
        x = self.joint_att(x, joint_text_embedding)

        feature_cls, feature_brd, div_loss = self.STI(x, mask, joint_text_embedding)
        self.div_loss = div_loss

        
        out_cls = self.conv_cls(feature_cls)
        out_bound = self.conv_bound(feature_brd)
        out_feature = self.conv_feature(feature_cls)
        out_feature_split = self.conv_feature_split(feature_cls+feature_brd)
        
        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]
            outputs_feature = [out_feature]

            for as_stage, conv_stage in zip(self.asb, self.conv_asb_feature):
                out_cls, feature_cls = as_stage(self.activation_asb(out_cls)* mask, feature_cls* mask, mask)
                out_feature = conv_stage(feature_cls)
                outputs_cls.append(out_cls)
                outputs_feature.append(out_feature)

            for br_stage in self.brb:
                out_bound,_ = br_stage(self.activation_brb(out_bound), mask)
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound, outputs_feature, out_feature_split, self.logit_scale)
        else:
            out_feature = None
            for as_stage, conv_stage in zip(self.asb, self.conv_asb_feature):
                out_cls, feature_cls = as_stage(self.activation_asb(out_cls)* mask, feature_cls* mask, mask)
                out_feature = conv_stage(feature_cls)

            bound_feature = None
            for br_stage in self.brb:
                out_bound, bound_feature = br_stage(self.activation_brb(out_bound), mask)

            return (out_cls, out_bound, out_feature, bound_feature)

class Model_hs(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_refine_layers: int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        SFI_layer: Optional[int] = None,
        dataset: str = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones(2) * np.log(1 / 0.07))  # 2.6593

        self.in_channel = in_channel
        node = 19 if dataset == "LARA" else 25

        self.SP_cls = MultiScale_GraphConv(13, in_channel, n_features, dataset) #MS-G3D
        self.SP_brd = MultiScale_GraphConv(13, in_channel, n_features, dataset) #MS-G3D
        self.STI_cls = STI(node, n_features, n_features, n_features, n_layers, SFI_layer)
        self.STI_brd = STI(node, n_features, n_features, n_features, n_layers, SFI_layer)
        self.joint_att_cls = Spatial_AttLayer(n_features, n_features, 32, 64, 4)
        self.joint_att_brd = Spatial_AttLayer(n_features, n_features, 32, 64, 4)

        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)
        self.conv_feature = nn.Conv1d(n_features,512,1)
        self.conv_feature_split = nn.Conv1d(n_features, 512, 1)

        # action segmentation branch
        asb = [
            copy.deepcopy(Decoder(n_classes, n_features, n_classes, n_refine_layers, alpha=exponential_descrease(s))) for s in range(n_stages_asb - 1)
        ]
        conv_asb_feature = [nn.Conv1d(n_features,512,1) for s in range(n_stages_asb - 1)]
        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_refine_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)
        self.conv_asb_feature = nn.ModuleList(conv_asb_feature)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()
        self.ff_embedding = nn.Linear(512, 512)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, joint_text_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        joint_text_embedding = self.ff_embedding(joint_text_embedding)
        joint_text_embedding = joint_text_embedding.permute(1, 0).unsqueeze(0).unsqueeze(2)
        # print(x.shape) #  torch.Size([3, 12, 19, 6000])

        x_cls = self.SP_cls(x) * mask.unsqueeze(3) #MS-G3D （n,c,t,v）
        x_brd = self.SP_brd(x) * mask.unsqueeze(3)
        # print(x.shape) #  torch.Size([3, 64, 6000, 19])
        # exit()
        x_cls = self.joint_att_cls(x_cls, joint_text_embedding)
        x_brd = self.joint_att_brd(x_brd, joint_text_embedding)

        feature_cls = self.STI_cls(x_cls, mask, joint_text_embedding)
        feature_brd = self.STI_brd(x_brd, mask, joint_text_embedding)

        
        out_cls = self.conv_cls(feature_cls)
        out_bound = self.conv_bound(feature_brd)
        out_feature = self.conv_feature(feature_cls)
        out_feature_split = self.conv_feature_split(feature_cls+feature_brd)
        
        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]
            outputs_feature = [out_feature]
            feature = feature_cls

            for as_stage, conv_stage in zip(self.asb, self.conv_asb_feature):
                out_cls, feature = as_stage(self.activation_asb(out_cls)* mask, feature* mask, mask)
                out_feature = conv_stage(feature)
                outputs_cls.append(out_cls)
                outputs_feature.append(out_feature)

            for br_stage in self.brb:
                out_bound,_ = br_stage(self.activation_brb(out_bound), mask)
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound, outputs_feature, out_feature_split, self.logit_scale)
        else:
            out_feature = None
            feature = feature_cls
            for as_stage, conv_stage in zip(self.asb, self.conv_asb_feature):
                out_cls, feature = as_stage(self.activation_asb(out_cls)* mask, feature* mask, mask)
                out_feature = conv_stage(feature)

            bound_feature = None
            for br_stage in self.brb:
                out_bound, bound_feature = br_stage(self.activation_brb(out_bound), mask)

            return (out_cls, out_bound, out_feature, bound_feature)



class Spatial_AttLayer(nn.Module):
    def __init__(self, in_channels, out_channels, qk_dim, v_dim, num_heads):
        super().__init__()
        self.qk_dim = qk_dim
        self.v_dim = v_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.query_conv = nn.Conv2d(in_channels=512, out_channels=num_heads * qk_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=512, out_channels=num_heads * qk_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels= in_channels, out_channels= v_dim, kernel_size=1)

        z = self.conv_out = nn.Conv2d(in_channels=num_heads * v_dim, out_channels=out_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, feature, text_feature):

        N, C, T, V = feature.size()

        k_feature = text_feature.expand(N, -1, 1, -1)
        q_feature = text_feature.expand(N, -1, 1, -1)
        v_feature = feature

        q = self.query_conv(q_feature).view(N, self.num_heads, self.qk_dim, V)
        k = self.key_conv(k_feature).view(N, self.num_heads, self.qk_dim, V)
        v = self.value_conv(v_feature)

        energy = torch.einsum('nhcu,nhcv->nhuv', q,k)
        attention = energy / (np.sqrt(self.qk_dim) * 1.0)
        attention = self.softmax(attention)

        z = torch.einsum('nctu,nhuv->nhctv', v, attention).contiguous().view(N, self.num_heads * self.v_dim, T, V)
        z = self.conv_out(z) + feature

        return z
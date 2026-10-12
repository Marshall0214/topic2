import torch
import torch.nn as nn
import torch.nn.functional as F

class labelTextEncoder(nn.Module): 
    def __init__(self, eps_std=0.0005, dim=2048, mu_size=None, joints=8, num=20):
        super(labelTextEncoder, self).__init__()
        # 对输入特征进行线性映射得到均值 μ
        self.mu_enc = nn.Linear(num, 1)
        # 利用一个两层 MLP 预测对数方差，包含 dropout 和 ReLU
        self.var_enc = nn.Sequential(
            nn.Linear(num, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.relu = nn.ReLU()
        self.eps_std = eps_std
        self.joints = joints
        self.num = num
        self.dim = dim
        # [19, 512, 1] -> [1, 512, 1, 19]
        

    def sample_gaussian_tensors(self, mu, logsigma, num_samples):
        # 生成采样噪声，形状为 [num_samples, B, T, D]
        eps = torch.normal(0, self.eps_std, size=(num_samples, *mu.size()), device=mu.device, dtype=mu.dtype)
        # 计算采样结果，使用 Einstein 求和进行逐元素乘法（这里与直接广播效果相同）
        samples = torch.einsum('nbtd,btd->nbtd', eps, torch.exp(0.5 * logsigma))
        # 调整采样结果维度：从 [num_samples, B, T, D] 变为 [B, num_samples, T, D]
        samples = samples.permute(1, 0, 2, 3)
        # 加上均值，得到最终采样结果
        samples = samples + mu.unsqueeze(1).repeat(1, num_samples, 1, 1)
        return samples.mean(1) 

    def forward(self, f, num_samples, is_training): 
        # 假设输入 f 形状为 [B, D, T]，调整为 [B, T, D]
        f = f.reshape(self.joints, self.num, -1)
        f = f.permute(0, 2, 1)
        # 预测均值 μ 和对数方差（经过 ReLU 限制）
        mu = self.mu_enc(f)
        var = self.relu(self.var_enc(f))
        # print(mu.shape, var.shape) # torch.Size([19, 512, 1]) torch.Size([19, 512, 1])
        # exit()
        if is_training:
            embedding = self.sample_gaussian_tensors(mu, var, num_samples)
            # assert embedding.shape == (self.joints, 1, 1, self.dim)
            # [19, 512, 1] -> [1, 512, 1, 19]

            # embedding = embedding.permute(1, 3, 2, 0)
            return mu.permute(2, 1, 0).unsqueeze(2), embedding.permute(2, 1, 0).unsqueeze(2), var.permute(2, 1, 0).unsqueeze(2) 
        else:
            # 推理模式下返回均值，同时复制一份作为采样结果
            return mu.permute(2, 1, 0).unsqueeze(2), mu.permute(2, 1, 0).unsqueeze(2), var.permute(2, 1, 0).unsqueeze(2)



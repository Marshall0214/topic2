import torch  

def diversity_loss(P, K):  
    """  
    计算多样性损失 Ldiv。支持二维 (K, Np) 或三维 (B, K, Np) 输入  
    
    参数:  
    - P: 形状为 (K, Np) 或 (B, K, Np) 的张量，表示 K 个关键点的属性嵌入  
    - K: int，关键点的数量  

    返回:  
    - loss: 计算出的多样性损失（标量）  
    """  
    # 转置最后两个维度以便矩阵乘法
    P_transposed = P.transpose(-1, -2)  # 形状变为 (..., Np, K)
    # 计算外积 P^T P
    P_outer = torch.matmul(P_transposed, P)  # 形状变为 (..., Np, Np)

    # 创建单位矩阵（自动适配设备）
    I = torch.eye(P.size(-1), device=P.device)  # 形状 (Np, Np)

    # 计算差异矩阵
    diff = P_outer - I  # 广播机制自动扩展维度

    # 根据输入维度选择计算方式
    if P.dim() == 3:
        # 三维输入时，逐样本计算范数
        loss_per_sample = torch.linalg.norm(diff, ord='fro', dim=(1,2)) ** 2  # 形状 (B,)
        loss = (loss_per_sample / K).mean()  # 先除以K再取平均
    else:
        # 二维输入保持原计算方式
        loss = torch.norm(diff, p='fro') ** 2 / K

    return loss

# 三维输入示例
B = 2  # 批量大小
K = 5  # 关键点数量
Np = 3 # 属性维度

P_3d = torch.randn(B, K, Np)  # 形状 (B, K, Np)
loss_3d = diversity_loss(P_3d, K)
print("3D Input Loss:", loss_3d.item())

# 二维输入示例（与原代码兼容）
P_2d = torch.randn(K, Np)     # 形状 (K, Np)
loss_2d = diversity_loss(P_2d, K)
print("2D Input Loss:", loss_2d.item())
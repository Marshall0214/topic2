import numpy as np
import torch
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def merge_by_segment(feats, labs):
    """
    按连续标签段合并特征：对每个连续相同标签的片段，计算特征均值作为该片段的代表特征
    
    参数:
        feats (np.ndarray or torch.Tensor): 每帧特征，形状为 (T, C)（T=帧数，C=特征维度）
        labs (np.ndarray or torch.Tensor): 每帧标签，形状为 (T,)
    
    返回:
        merged_feats (np.ndarray): 合并后的片段特征，形状为 (S, C)（S=片段数）
        merged_labs (np.ndarray): 合并后的片段标签，形状为 (S,)
    """
    # 1. 类型转换：torch.Tensor → numpy.ndarray（避免设备不一致问题）
    if isinstance(feats, torch.Tensor):
        feats = feats.cpu().numpy()
    if isinstance(labs, torch.Tensor):
        labs = labs.cpu().numpy()
    
    # 2. 输入合法性检查
    assert feats.ndim == 2, f"特征需为2维(T, C)，当前为{feats.ndim}维"
    assert labs.ndim == 1, f"标签需为1维(T,)，当前为{labs.ndim}维"
    assert feats.shape[0] == labs.shape[0], f"特征帧数({feats.shape[0]})与标签数({labs.shape[0]})不匹配"
    
    # 3. 处理空输入（避免后续循环报错）
    if len(labs) == 0:
        print("警告：输入数据为空，返回空数组")
        return np.array([]), np.array([])
    
    # 4. 按连续标签段合并特征
    merged_feats = []
    merged_labs = []
    start = 0  # 每个片段的起始索引
    
    # 遍历标签寻找片段边界（相邻标签不同则分割）
    for i in range(1, len(labs)):
        if labs[i] != labs[i - 1]:
            # 计算当前片段（start~i-1）的特征均值
            segment_mean = feats[start:i].mean(axis=0)
            merged_feats.append(segment_mean)
            merged_labs.append(labs[i - 1])
            start = i  # 更新下一个片段的起始索引
    
    # 处理最后一个片段（start~序列末尾）
    last_segment_mean = feats[start:].mean(axis=0)
    merged_feats.append(last_segment_mean)
    merged_labs.append(labs[-1])
    
    # 转换为numpy数组并返回
    return np.stack(merged_feats, axis=0), np.array(merged_labs)


# def plot_combined_tsne(features_list, labels_list, save_path='tsne_combined.png', npy_prefix='tsne_data'):
#     np_feats = []
#     np_labels = []
#     for feat, lab in zip(features_list, labels_list):
#         if isinstance(feat, torch.Tensor):
#             feat = feat.cpu().numpy()
#         if isinstance(lab, torch.Tensor):
#             lab = lab.cpu().numpy()
#         assert feat.ndim == 2 and lab.ndim == 1 and feat.shape[0] == lab.shape[0]
#         mask = lab != 0
#         feat = feat[mask]
#         lab = lab[mask]
#         feat, lab = merge_by_segment(feat, lab)
#         np_feats.append(feat)
#         np_labels.append(lab)

#     all_feats = np.concatenate(np_feats, axis=0)
#     all_labels = np.concatenate(np_labels, axis=0)
#     plot_tsne(all_feats, all_labels, save_path=save_path, title='Combined t-SNE Visualization')

#     npy_dir = os.path.dirname(save_path) or '.'
#     np.save(os.path.join(npy_dir, f'{npy_prefix}_features.npy'), all_feats)
#     np.save(os.path.join(npy_dir, f'{npy_prefix}_labels.npy'), all_labels)

#     print(f"[+] 特征已保存到: {npy_prefix}_features.npy")
#     print(f"[+] 标签已保存到: {npy_prefix}_labels.npy")


def plot_combined_tsne(features_list, labels_list, save_path='tsne_combined.png', npy_prefix='tsne_data'):
    """
    将多条序列特征拼接后绘制全局 t-SNE，并保存为 .npy 便于对比可视化

    参数:
        features_list (list of Tensor/ndarray): 多条序列的 feature_tensor 列表
        labels_list   (list of Tensor/ndarray): 多条序列的 label_tensor 列表
        save_path (str): t-SNE 图保存路径
        npy_prefix (str): 保存 npy 的文件前缀（不含扩展名）
    """
    # 转列表中的所有为 numpy，并拼接
    np_feats = []
    np_labels = []
    for feat, lab in zip(features_list, labels_list):
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().numpy()
        if isinstance(lab, torch.Tensor):
            lab = lab.cpu().numpy()
        assert feat.ndim == 2 and lab.ndim == 1 and feat.shape[0] == lab.shape[0]
        np_feats.append(feat)
        np_labels.append(lab)

    all_feats = np.concatenate(np_feats, axis=0)   # (sum T_i, C)
    all_labels = np.concatenate(np_labels, axis=0) # (sum T_i,)

    # ✅ 保存 .npy 文件
    print("Saving features and labels to .npy files...")
    npy_dir = os.path.dirname(save_path) or '.'
    np.save(os.path.join(npy_dir, f'{npy_prefix}_features.npy'), all_feats)
    np.save(os.path.join(npy_dir, f'{npy_prefix}_labels.npy'), all_labels)

    # 调用绘图函数
    print("Plotting t-SNE...")
    # plot_tsne(all_feats, all_labels, save_path=save_path, title='Combined t-SNE Visualization')
    
    print(f"共有{all_feats.shape[0]}帧特征，已绘制 t-SNE 图：{save_path}")
    print(f"[+] 特征已保存到: {npy_prefix}_features.npy")
    print(f"[+] 标签已保存到: {npy_prefix}_labels.npy")

# def plot_combined_tsne(features_list, labels_list, save_path='tsne_combined.png'):
#     """
#     将多条序列特征拼接后绘制全局 t-SNE

#     参数:
#         features_list (list of Tensor/ndarray): 多条序列的 feature_tensor 列表
#         labels_list (list of Tensor/ndarray): 多条序列的 label_tensor 列表
#         save_path (str): 保存路径
#     """
#     # 转列表中的所有为 numpy，并拼接
#     np_feats = []
#     np_labels = []
#     for feat, lab in zip(features_list, labels_list):
#         if isinstance(feat, torch.Tensor):
#             feat = feat.cpu().numpy()
#         if isinstance(lab, torch.Tensor):
#             lab = lab.cpu().numpy()
#         assert feat.ndim == 2 and lab.ndim == 1 and feat.shape[0] == lab.shape[0]
#         np_feats.append(feat)
#         np_labels.append(lab)

#     all_feats = np.concatenate(np_feats, axis=0)   # (sum T_i, C)
#     all_labels = np.concatenate(np_labels, axis=0) # (sum T_i,)

#     plot_tsne(all_feats, all_labels, save_path=save_path, title='Combined t-SNE Visualization')

import numpy as np
import torch
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def sample_middle_segment_feats(feats, labs, keep_ratio=0.1):
    """
    从每个连续动作片段中，截取中间部分的一定比例帧特征。

    参数:
    ----
    feats : (T, C) ndarray
        每帧特征
    labs  : (T,) ndarray
        每帧标签
    keep_ratio : float
        每个片段保留的比例（默认 0.1，即 10%）

    返回:
    ----
    sampled_feats : (S, C) ndarray
    sampled_labs  : (S,) ndarray
    """
    assert feats.shape[0] == labs.shape[0]
    sampled_feats, sampled_labs = [], []

    start = 0
    for i in range(1, len(labs)):
        if labs[i] != labs[i - 1]:
            segment_feats = feats[start:i]
            segment_labs = labs[start:i]
            L = len(segment_labs)
            keep = max(1, int(L * keep_ratio))  # 至少保留1帧
            mid_start = (L - keep) // 2
            mid_end = mid_start + keep
            sampled_feats.append(segment_feats[mid_start:mid_end])
            sampled_labs.append(segment_labs[mid_start:mid_end])
            start = i

    # 处理最后一个片段
    segment_feats = feats[start:]
    segment_labs = labs[start:]
    L = len(segment_labs)
    keep = max(1, int(L * keep_ratio))
    mid_start = (L - keep) // 2
    mid_end = mid_start + keep
    sampled_feats.append(segment_feats[mid_start:mid_end])
    sampled_labs.append(segment_labs[mid_start:mid_end])

    return np.concatenate(sampled_feats, axis=0), np.concatenate(sampled_labs, axis=0)


def save_middle_sampled_tsne(features_list, labels_list,
                             save_path='tsne_middle_sampled.png',
                             npy_prefix='tsne_middle_sampled',
                             keep_ratio=0.1,
                             perplexity=30):
    """
    对多条序列进行中段抽样，然后绘制和保存 t-SNE。

    参数:
    ----
    features_list : list[np.ndarray or torch.Tensor]
    labels_list   : list[np.ndarray or torch.Tensor]
    save_path     : str
        保存 t-SNE 图片的路径
    npy_prefix    : str
        中间结果 npy 保存前缀
    keep_ratio    : float
        每个片段中保留帧比例
    perplexity    : int
        t-SNE 参数
    """
    np_feats, np_labels = [], []

    for feat, lab in zip(features_list, labels_list):
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().numpy()
        if isinstance(lab, torch.Tensor):
            lab = lab.cpu().numpy()
        assert feat.ndim == 2 and lab.ndim == 1 and feat.shape[0] == lab.shape[0]

        # 过滤掉 label=0
        mask = lab != 0
        feat = feat[mask]
        lab = lab[mask]

        # 中段抽样
        sampled_feat, sampled_lab = sample_middle_segment_feats(feat, lab, keep_ratio)
        np_feats.append(sampled_feat)
        np_labels.append(sampled_lab)

    all_feats = np.concatenate(np_feats, axis=0)
    all_labels = np.concatenate(np_labels, axis=0)

    # 保存中间文件
    npy_dir = os.path.dirname(save_path) or '.'
    os.makedirs(npy_dir, exist_ok=True)
    np.save(os.path.join(npy_dir, f'{npy_prefix}_features.npy'), all_feats)
    np.save(os.path.join(npy_dir, f'{npy_prefix}_labels.npy'), all_labels)

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    feats_2d = tsne.fit_transform(all_feats)

    # 绘图
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(feats_2d[:, 0], feats_2d[:, 1],
                     c=all_labels, cmap='tab20', s=5, alpha=0.8)
    plt.colorbar(sc, label='Class')
    plt.title(f't-SNE (Middle {int(keep_ratio*100)}% Sampling)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[+] 保存 t-SNE 图: {save_path}")
    print(f"[+] 特征保存到: {npy_prefix}_features.npy")
    print(f"[+] 标签保存到: {npy_prefix}_labels.npy")


# 示例用法
if __name__ == "__main__":
    """
    假设文件目录结构：
    exp/
        model_A_features.npy   (N, C)
        model_A_labels.npy     (N,)
        model_B_features.npy   (M, C)
        model_B_labels.npy     (M,)
    注意：N 和 M 可以不同，内部会自动过滤 label==0 并统一 t-SNE。
    """
    # pku sub
    # rootA = '/gpu02_data/DUANXINRAN/results_log_2025_08_25/result_fwag/cross4/PKU-subject/split1/test_newplot/tsne'
    # rootB = '/gpu02_data/DUANXINRAN/results_log_2025_08_25/result_fwag/origin/PKU-subject/split1/test_newplot/tsne'                        # 换成你的路径
    # LARA
    rootA = '/gpu02_data/DUANXINRAN/results_log_2025_09_05/result_tune_Gnum/cross1/LARA/split1/test_newplot/tsne'
    rootB = '/DUANXINRAN/LaSA/results/origin/LARA/split1/test_newplot/tsne'                        # 换成你的路径
    # featA_path = os.path.join(rootA, 'tsne_data_features.npy')
    # labA_path  = os.path.join(rootA, 'tsne_data_labels.npy')
    # featB_path = os.path.join(rootB, 'tsne_data_features.npy')
    # labB_path  = os.path.join(rootB, 'tsne_data_labels.npy')
    featA_path = os.path.join(rootA, 'tsne_middle_sampled_features.npy')
    labA_path  = os.path.join(rootA, 'tsne_middle_sampled_labels.npy')
    featB_path = os.path.join(rootB, 'tsne_middle_sampled_features.npy')
    labB_path  = os.path.join(rootB, 'tsne_middle_sampled_labels.npy')

    # 加载
    featA = np.load(featA_path)         # (N, C)
    labA  = np.load(labA_path)          # (N,)
    featB = np.load(featB_path)         # (M, C)
    labB  = np.load(labB_path)          # (M,)

    # 为了演示，这里仅把单条序列放进 list；如果你有多个片段，可继续 append
    featsA_list = [torch.from_numpy(featA)]
    labsA_list  = [torch.from_numpy(labA)]
    featsB_list = [torch.from_numpy(featB)]
    labsB_list  = [torch.from_numpy(labB)]

    # 统一 t-SNE 并绘图
    os.makedirs('outputs', exist_ok=True)
    plot_dual_tsne(featsA_list, labsA_list,
                   featsB_list, labsB_list,
                   save_prefix=os.path.join(rootA, 'tsne_compare_AB.png'),
                   perplexity=30,
                   filter_zero=True)
    print("Done! 已生成 A/B 对比图：", os.path.join(rootA, 'tsne_compare_AB.png'))

# def plot_tsne(feature_tensor, label_tensor, save_path='tsne_plot.png'):
#     """
#     绘制 t-SNE 图并保存
    
#     参数:
#         feature_tensor (Tensor): 输入的特征张量，形状为 (T, C)
#         label_tensor (Tensor): 输入的标签张量，形状为 (T,)
#         save_path (str): 保存 t-SNE 图的路径，默认为 'tsne_plot.png'
#     """
#     # 确保输入是 NumPy 数组
#     if isinstance(feature_tensor, torch.Tensor):
#         feature_tensor = feature_tensor.cpu().numpy()
#     if isinstance(label_tensor, torch.Tensor):
#         label_tensor = label_tensor.cpu().numpy()
    
#     # 检查输入形状
#     assert feature_tensor.ndim == 2, "特征张量必须是二维的 (T, C)"
#     assert label_tensor.ndim == 1, "标签张量必须是一维的 (T,)"
#     assert feature_tensor.shape[0] == label_tensor.shape[0], "特征和标签的样本数必须一致"
    
#     # t-SNE 降维
#     tsne = TSNE(n_components=2, random_state=42)
#     features_2d = tsne.fit_transform(feature_tensor)
    
#     # 绘制 t-SNE 图
#     plt.figure(figsize=(10, 8))
#     scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=label_tensor, cmap='viridis', s=5)
#     plt.colorbar(scatter, label='Label')
#     plt.title('t-SNE Visualization')
#     plt.xlabel('t-SNE 1')
#     plt.ylabel('t-SNE 2')
    
#     # 确保保存路径的目录存在
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
#     # 保存图像
#     plt.savefig(save_path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"t-SNE 图已保存到: {save_path}")

# # 示例用法
# if __name__ == "__main__":
#     # 生成随机数据进行测试
#     T, C = 1000, 128
#     feature_tensor = torch.randn(T, C)
#     label_tensor = torch.randint(0, 10, (T,))
    
#     # 调用函数绘制 t-SNE 图

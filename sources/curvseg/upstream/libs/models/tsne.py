import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import torch



import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import torch

def plot_dual_tsne(featsA_list, labsA_list,
                   featsB_list, labsB_list,
                   save_prefix='tsne_dual',
                   perplexity=30,
                   filter_zero=True):
    """
    对两个模型的特征序列使用同一套 t-SNE 映射，并排可视化并保存。

    参数
    ----
    featsA_list / featsB_list : list[Tensor/ndarray]
        每个元素形状 (T_i, C)，对应一条序列的特征。
    labsA_list / labsB_list   : list[Tensor/ndarray]
        每条序列的标签，与 feats 逐条对齐。
    save_prefix   : str
        保存文件前缀，最终保存 <prefix>_A.png, <prefix>_B.png, <prefix>_side_by_side.png
    perplexity    : int
        t-SNE 的 perplexity 参数。
    filter_zero   : bool
        是否过滤标签为 0 的帧。
    """

    # ---------- 1. 合并并转为 numpy ----------
    def _to_np(x):
        return x.cpu().numpy() if isinstance(x, torch.Tensor) else x

    def _concat(list_feats, list_labs):
        feats_np = [_to_np(f) for f in list_feats]
        labs_np  = [_to_np(l) for l in list_labs]
        # 过滤 label=0
        if filter_zero:
            feats_np = [f[l != 0] for f, l in zip(feats_np, labs_np)]
            labs_np  = [l[l != 0] for l in labs_np]
        feats_all = np.concatenate(feats_np, axis=0)
        labs_all  = np.concatenate(labs_np,  axis=0)
        return feats_all, labs_all, [len(f) for f in feats_np]   # 返回长度用于拆分

    featsA, labsA, lenA = _concat(featsA_list, labsA_list)
    featsB, labsB, lenB = _concat(featsB_list, labsB_list)

    # ---------- 2. 统一 t-SNE ----------
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    feats_all = np.concatenate([featsA, featsB], axis=0)
    emb_all   = tsne.fit_transform(feats_all)

    # ---------- 3. 拆分 ----------
    embA = emb_all[:len(featsA)]
    embB = emb_all[len(featsA):]

    # ---------- 4. 辅助绘图 ----------
    def _plot(emb, labs, title, save_path):
        plt.figure(figsize=(6, 5))
        sc = plt.scatter(emb[:, 0], emb[:, 1],
                         c=labs, cmap='tab10', s=4, alpha=0.8)
        plt.colorbar(sc, label='Label')
        plt.title(title)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved: {save_path}")

    # ---------- 5. 保存 ----------
    base = os.path.splitext(save_prefix)[0]
    _plot(embA, labsA, title="Ours", save_path=f"{base}_A.png")
    _plot(embB, labsB, title="LaSA", save_path=f"{base}_B.png")

    # ---------- 6. 并排图 ----------
    plt.figure(figsize=(24, 10))
    plt.subplot(1, 2, 1)
    sc = plt.scatter(embA[:, 0], embA[:, 1],
                     c=labsA, cmap='tab10', s=4, alpha=0.8)
    plt.colorbar(sc, label='Label')
    plt.title("Ours Features")

    plt.subplot(1, 2, 2)
    sc = plt.scatter(embB[:, 0], embB[:, 1],
                     c=labsB, cmap='tab10', s=4, alpha=0.8)
    plt.colorbar(sc, label='Label')
    plt.title("LaSA Features")
    plt.tight_layout()
    plt.savefig(f"{base}_side_by_side.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {base}_side_by_side.png")

def plot_tsne(feature_tensor, label_tensor, save_path='tsne_plot.png', title='t-SNE Visualization'):
    """
    绘制单条序列的 t-SNE 图并保存

    参数:
        feature_tensor (Tensor or np.ndarray): 特征张量，形状 (T, C)
        label_tensor (Tensor or np.ndarray): 标签张量，形状 (T,)
        save_path (str): 保存路径
        title (str): 图标题
    """
    # 转 NumPy
    if isinstance(feature_tensor, torch.Tensor):
        feature_tensor = feature_tensor.cpu().numpy()
    if isinstance(label_tensor, torch.Tensor):
        label_tensor = label_tensor.cpu().numpy()
    
    mask = label_tensor != 0
    feature_tensor = feature_tensor[mask]
    label_tensor   = label_tensor[mask]

    # 检查形状
    assert feature_tensor.ndim == 2, "feature_tensor 必须为 (T, C)"
    assert label_tensor.ndim == 1, "label_tensor 必须为 (T,)"
    assert feature_tensor.shape[0] == label_tensor.shape[0], "样本数量不一致"

    # t-SNE 降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(feature_tensor)

    # 绘图
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                     c=label_tensor, cmap='viridis', s=5, alpha=0.8)
    plt.colorbar(sc, label='Label')
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')

    # 保存
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {save_path}")

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
    npy_dir = os.path.dirname(save_path) or '.'
    np.save(os.path.join(npy_dir, f'{npy_prefix}_features.npy'), all_feats)
    np.save(os.path.join(npy_dir, f'{npy_prefix}_labels.npy'), all_labels)

    # 调用绘图函数
    plot_tsne(all_feats, all_labels, save_path=save_path, title='Combined t-SNE Visualization')

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

    rootA = '/home/dxr/LaSA/result/drop005_brd+cur_th01_wofw_cross2/PKU-view/split1/test_th012/tsne'
    rootB = '/home/dxr/LaSA/result/dropout+0.05_origin/PKU-view/split1/test_th012/tsne'                        # 换成你的路径
    featA_path = os.path.join(rootA, 'tsne_data_features.npy')
    labA_path  = os.path.join(rootA, 'tsne_data_labels.npy')
    featB_path = os.path.join(rootB, 'tsne_data_features.npy')
    labB_path  = os.path.join(rootB, 'tsne_data_labels.npy')

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
#     plot_tsne(feature_tensor, label_tensor, save_path='tsne_example.png')
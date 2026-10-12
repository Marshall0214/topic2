import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Graph:
    def __init__(self, layout='MCFS-22'):
        self.get_edge(layout)

    def get_edge(self, layout):
        if layout == 'MCFS-22' or layout == 'MCFS-130':
            self.num_node = 25
            inward_ori_index = [
                (2, 9), (1, 2), (16, 1), (18, 16), (17, 1), (19, 17), (6, 2),
                (7, 6), (8, 7), (3, 2), (4, 3), (5, 4), (10, 9),
                (11, 10), (12, 11), (25, 12), (23, 12), (24, 23), (13, 9),
                (14, 13), (15, 14), (22, 15), (20, 15), (21, 20)
            ]
            self.inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
            self.outward = [(j, i) for (i, j) in self.inward]
            self.neighbor = self.inward + self.outward

def plot_skeleton_frame(ax, keypoints, edges, color='black'):
    for i, j in edges:
        x = [keypoints[i, 0], keypoints[j, 0]]
        y = [keypoints[i, 1], keypoints[j, 1]]
        ax.plot(x, y, color=color, linewidth=2)
    ax.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=10)

def normalize_keypoints(keypoints):
    """
    Normalize keypoints to be centered and scaled to [-0.5, 0.5] in each axis.
    keypoints: (25, 2) or (25, 3)
    """
    center = np.mean(keypoints, axis=0)
    keypoints = keypoints - center  # 中心化
    scale = np.max(np.linalg.norm(keypoints, axis=1))  # 最大骨架长度
    if scale > 0:
        keypoints = keypoints / scale  # 归一化
    return keypoints

def normalize_keypoints_01(keypoints):
    """
    将 keypoints 归一化到 [0, 1] 范围。
    keypoints: (25, 2) 或 (25, 3)
    返回：归一化后的 keypoints，形状不变
    """
    min_vals = np.min(keypoints, axis=0)
    max_vals = np.max(keypoints, axis=0)
    range_vals = max_vals - min_vals
    # 避免除以0
    range_vals[range_vals == 0] = 1e-6
    return (keypoints - min_vals) / range_vals


def visualize_skeleton_sequence(sequence, layout='MCFS-22', save_path=None):
    """
    sequence: numpy array of shape (T, 25, 2) or (T, 25, 3)
    save_path: either None (show), a file (e.g. xxx.mp4), or a directory (save each frame as image)
    """
    graph = Graph(layout)
    num_frames = sequence.shape[0]
    is_3d = sequence.shape[2] == 3

    for frame_idx in range(num_frames):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d' if is_3d else None)
        keypoints = sequence[frame_idx]
        keypoints = normalize_keypoints_01(keypoints)

        if is_3d:
            for i, j in graph.neighbor:
                x = [keypoints[i, 0], keypoints[j, 0]]
                y = [keypoints[i, 1], keypoints[j, 1]]
                z = [keypoints[i, 2], keypoints[j, 2]]
                ax.plot(x, y, z, color='blue', linewidth=2)
            ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], c='red', s=10)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
        else:
            plot_skeleton_frame(ax, keypoints, graph.neighbor)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

        ax.set_title(f'Frame {frame_idx + 1}/{num_frames}')
        plt.tight_layout()

        if save_path is None:
            plt.show()
        elif os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)
            frame_file = os.path.join(save_path, f"frame_{frame_idx:03d}.png")
            plt.savefig(frame_file)
            print(f"Saved frame to {frame_file}")
        else:
            # save as video (only once, not frame by frame)
            if frame_idx == 0:
                imgs = []

            buf = os.path.join("/tmp", f"temp_frame_{frame_idx}.png")
            plt.savefig(buf)
            imgs.append(buf)

            if frame_idx == num_frames - 1:
                from PIL import Image
                frames = [Image.open(p) for p in imgs]
                frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
                print(f"Saved animation to {save_path}")
                # cleanup
                for f in imgs:
                    os.remove(f)

        plt.close(fig)

# 示例用法
if __name__ == "__main__":
    import os

    # 假设你有一个 npy 文件存储骨架数据
    data_path = '/DUANXINRAN/LaSA/dataset/MCFS-22/features/demo.npy'  # shape: [T, 25, 2]
    if os.path.exists(data_path):
        skeleton_seq = np.load(data_path)
    else:
        # 生成假数据用于演示
        T = 60
        skeleton_seq = np.random.rand(T, 25, 2)
    skeleton_seq = skeleton_seq[:, :, :2]  # 确保是 2D 数据
    visualize_skeleton_sequence(skeleton_seq, layout='MCFS-22', save_path='/DUANXINRAN/LaSA/test')

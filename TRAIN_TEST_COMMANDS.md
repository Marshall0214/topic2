# 三个模型的训练、续训与测试命令

更新：2026-10-12。适用 Windows PowerShell，每个代码块中的命令均为单行。

本文的三个模型为因果 MS-TCN、骨架输入 ProTAS（官方固定任务图版本）和原版离线 SpecScalpel。命令使用现有本地适配代码，不代表原论文实验完全复现。

## 1. 环境与统一协议

先切换目录（同一个 PowerShell 窗口只需执行一次）：

```powershell
Set-Location 'D:\AQA\online_skeleton_segmentation_baselines'
```

Python：`C:\Users\admin\anaconda3\envs\dest_gpu\python.exe`。已安装 torch 2.8.0+cu126、torchvision 0.23.0+cu126；不需要再激活 conda。数据入口：`D:\AQA\online_skeleton_action_segmentation\official_data\dataset\PKU-subject`。

统一划分为 `configs/dest_compat.json` 中的 **698 条训练 / 77 条验证 / 234 条测试**。不合并 trainval。所有模型按验证集原始输出的 **(canonical Edit + F1@50)/2** 选最佳权重；分数相同时保留较早轮次。测试时指定 `best`，不能根据测试分数改选 `last` 或其他轮次。

| 模型 | 定位 | 当前训练预算 | 批大小 | 学习率 |
|---|---|---:|---:|---:|
| MS-TCN | 因果骨架适配基线 | 50 轮 | 1 | 0.0005 |
| ProTAS | 在线骨架适配基线，固定任务图 | 50 轮 | 1 | 0.0005 |
| SpecScalpel | 离线骨架与文本先验参照 | 300 轮 | 8 | 0.001 |

前两者使用统一相对坐标、后向位移特征；SpecScalpel 保留原版相对坐标、前向位移及文本先验。统一的是划分、选模规则与评价器，**不是相同输入变换、训练预算或未来信息条件**，结果不能宣称为完全同条件的算法排名。SpecScalpel 的独立测试入口同时报告原始输出与官方边界后处理输出；主选模仍依据原始输出。

新训练命令不会覆盖已有实验目录。前两个复现命令使用新名字；SpecScalpel 使用此前约定的正式实验名字。已有同名目录时，不要再次执行从头训练命令。

## 2. 因果 MS-TCN

### 从头训练：完整训练集，50 轮

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/train_small.py --model mstcn --run mstcn_full_repro_seed17 --split configs/dest_compat.json --epochs 50 --train-limit 0 --val-limit 0 --clip 0 --seed 17 --device cuda --stages 4 --layers 10 --width 64 --lr 0.0005 --normalization none
```

`train_small.py` 虽然名字包含 small，但 `--train-limit 0 --val-limit 0 --clip 0` 表示全部训练、验证数据和完整序列。

### 中断后继续：总共训练至第 50 轮

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/train_small.py --model mstcn --run mstcn_full_repro_resume_seed17 --resume-from mstcn_full_repro_seed17 --split configs/dest_compat.json --epochs 50 --train-limit 0 --val-limit 0 --clip 0 --seed 17 --device cuda --stages 4 --layers 10 --width 64 --lr 0.0005 --normalization none
```

该入口会读取原实验的最后权重与优化器状态，将续训结果写入**新目录**。不是额外训练 50 轮。它不保存完整随机状态，不应声称与未中断训练逐步完全一致。若已经完成 50 轮，不需要执行此命令。

### 验证最佳权重

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/evaluate_run.py --run mstcn_full_repro_seed17 --model mstcn --checkpoint best --split val --device cuda
```

### 测试最佳权重：训练、选模完成后执行

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/evaluate_run.py --run mstcn_full_repro_seed17 --model mstcn --checkpoint best --split test --device cuda
```

如果实际使用了续训结果，将上面两条评价命令的 run 改为 `mstcn_full_repro_resume_seed17`。

权重：`runs/<run>/mstcn/best.pt`、`last.pt`。训练记录：`runs/<run>/mstcn/history.json`。测试结果：`runs/<run>/test_best_evaluation.json`。

## 3. 骨架版 ProTAS：官方固定任务图

### 从头训练：完整训练集，50 轮

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/train_small.py --model protas --run protas_fixedgraph_full_repro_seed17 --split configs/dest_compat.json --epochs 50 --train-limit 0 --val-limit 0 --clip 0 --seed 17 --device cuda --stages 4 --layers 10 --width 64 --lr 0.0005 --normalization none --graph-reference official --task-graph fixed
```

必须保留 `--graph-reference official --task-graph fixed`。本文主基线为恢复官方固定任务图行为的版本，不使用早期的可学习任务图版本。任务图由训练标签构建，不使用验证或测试标签。

### 中断后继续：总共训练至第 50 轮

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/train_small.py --model protas --run protas_fixedgraph_full_repro_resume_seed17 --resume-from protas_fixedgraph_full_repro_seed17 --split configs/dest_compat.json --epochs 50 --train-limit 0 --val-limit 0 --clip 0 --seed 17 --device cuda --stages 4 --layers 10 --width 64 --lr 0.0005 --normalization none --graph-reference official --task-graph fixed
```

同 MS-TCN，续训写入新目录，并恢复最后完成的轮次、模型和优化器状态；不承诺完整随机过程一致。

### 验证最佳权重

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/evaluate_run.py --run protas_fixedgraph_full_repro_seed17 --model protas --checkpoint best --split val --device cuda
```

### 测试最佳权重：训练、选模完成后执行

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/evaluate_run.py --run protas_fixedgraph_full_repro_seed17 --model protas --checkpoint best --split test --device cuda
```

如果实际使用了续训结果，将两条评价命令的 run 改为 `protas_fixedgraph_full_repro_resume_seed17`。

权重：`runs/<run>/protas/best.pt`、`last.pt`。训练记录：`runs/<run>/protas/history.json`。测试结果：`runs/<run>/test_best_evaluation.json`。

## 4. SpecScalpel：原版离线参照

### 从头训练：完整训练集，300 轮

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/verify_specscalpel_training.py --run specscalpel_full_seed17 --train-limit 0 --val-limit 0 --batch-size 8 --epochs 300 --seed 17
```

此正式目录已经在本地存在。如果该训练已开始，应继续现有进程；若中断，用下方续训命令。不要删除已有实验重跑。脚本名虽然包含 verify，但现在支持全量训练。

保留官方模型、完整损失、文本先验、随机打乱和 batch size 8。全量训练时按官方行为丢弃不足一批的尾批，每轮实际参与 696 条序列，打乱后被丢弃的序列可以不同；整个训练候选集合仍为 698 条。当前 num_workers=0 以兼容 Windows，数值兼容适配仅替换边界后处理中的旧 NumPy 布尔类型。

### 中断后继续：总共训练至第 300 轮

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/verify_specscalpel_training.py --run specscalpel_full_seed17 --train-limit 0 --val-limit 0 --batch-size 8 --epochs 300 --seed 17 --resume
```

SpecScalpel 的续训直接使用**同一个目录**，恢复 last.pt 中的模型、优化器、历史与随机状态。从最后一个完整完成的训练并验证轮次之后继续，中途未保存的一轮会重做。断点续训入口已验证通过，但不保证所有 CUDA 算子逐位确定性。

### 验证最佳权重

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/evaluate_specscalpel.py --run specscalpel_full_seed17 --checkpoint best --split val
```

### 测试最佳权重：300 轮训练、选模完成后执行

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/evaluate_specscalpel.py --run specscalpel_full_seed17 --checkpoint best --split test
```

权重：`runs/specscalpel_full_seed17/best.pt`、`last.pt`。逐轮记录：`history.json`。测试结果：`test_best_evaluation.json`。其中 `raw` 为原始类别输出，`boundary_refined` 为官方边界后处理输出；`foreground_canonical` 为主统一口径，`dest_legacy` 和 `protas_official` 为现有补充口径，不应误称为 SpecScalpel 作者评测器。

## 5. 已完成的旧实验：无需重新训练

如果只是重新评价已经完成的两个基线，使用原目录即可：

MS-TCN 已有实验测试：

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/evaluate_run.py --run mstcn_pku_dev_seed17 --model mstcn --checkpoint best --split test --device cuda
```

ProTAS 官方固定图已有实验测试：

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/evaluate_run.py --run protas_pku_fixedgraph_dev_seed17 --model protas --checkpoint best --split test --device cuda
```

上述命令会重新计算并写入对应测试 JSON；不需要仅为了生成本命令文档而执行。MS-TCN 的 100 轮扩展实验和 ProTAS 早期可学习图实验属于其他实验，不与上述目录混用。

## 6. 执行顺序与状态解释

2026-10-12 收敛复核补充：MS-TCN 已有 `mstcn_pku_dev_e100_resume_seed17` 的 100 轮实验，最佳验证轮次为 62。固定图 ProTAS 尚只有 50 轮；建议待当前 GPU 训练完成后续训到 100 轮，再作同预算在线比较。上文 50 轮命令保留用于复现历史实验。下一次计划的续训命令及证据见 `reports/online_convergence_review/REPORT.md`，不需要重跑 MS-TCN 的同一 100 轮实验。

一次运行一个 GPU 训练命令。首次训练使用从头命令，中断才使用续训命令；两者是替代关系，不是连续执行的两步。完成训练后，再显式运行对应测试命令。

`--epochs` 表示总目标轮数，`--train-limit 0` / `--val-limit 0` 表示全量，不是零条数据。训练过程中的 Acc、Edit、F1 是验证集结果；执行 `--split test` 后生成的文件才是测试结果。单独观察 Acc 可能受到背景比例影响，应同时检查前景准确率和 F1@50。

本次仅用已有检查实验验证新 SpecScalpel 评价入口的验证集路径，没有启动正式训练，也没有执行任何测试集评价。

# 在线基线收敛与改进目标核查

2026-10-12。只读取训练与验证日志、配置和已有验证错误报告；未启动 GPU 训练、未运行模型推理、未访问测试集。与正在进行的 SpecScalpel 训练不争抢 GPU。

## 1. 结论先行

MS-TCN 已有 100 轮结果，不需要重新安排相同实验；固定图 ProTAS 目前仅完成 50 轮。应先将固定图 ProTAS 补至 100 轮，随后按相同总预算重新比较两个在线基线并更新验证错误分析，再确定算法改进。续训可在 SpecScalpel 完成后执行。

原来的“两个模型目前均只有 50 轮”表述不准确；旧 50 轮对照可保留为历史同预算实验，但不足以代表目前最充分训练的 MS-TCN。

## 2. 同一验证集下的已保存结果

| 模型 | 总训练轮数 | 选中轮次 | canonical Edit | F1@50 | 选模分数 |
|---|---:|---:|---:|---:|---:|
| 因果 MS-TCN | 50 | 50 | 47.29 | 33.79 | 40.54 |
| 因果 MS-TCN 续训 | 100 | 62 | 49.88 | 39.01 | 44.45 |
| 固定图 ProTAS | 50 | 46 | 52.30 | 37.07 | 44.68 |

选模分数为 `(Edit + F1@50)/2`。MS-TCN 延长训练后的验证 F1@50 提高约 5.22 个百分点，说明 50 轮不是充分的截止依据。固定图 ProTAS 的分数 44.68 与 MS-TCN 的 44.45 来自不同训练预算，不能据此作严格同预算优劣判断；也没有多 seed 统计支持。

代码复核三个配置的 split SHA256 与实际 selected train/val 列表相同，均为 698/77 的开发协议；MS-TCN 100 轮日志的前 50 轮与原日志逐项一致，确认它是同一实验的续训历史，而非独立重跑。

## 3. 收敛迹象

下表是每 10 轮窗口的平均验证选模分数，不能把窗口均值当作一次独立实验或统计显著性证据。

| 轮次窗口 | MS-TCN | 固定图 ProTAS |
|---|---:|---:|
| 21–30 | 33.75 | 35.94 |
| 31–40 | 35.40 | 39.71 |
| 41–50 | 37.77 | 40.17 |
| 51–60 | 39.61 | 未训练 |
| 61–70 | 40.41 | 未训练 |
| 71–80 | 38.76 | 未训练 |
| 81–90 | 38.86 | 未训练 |
| 91–100 | 38.43 | 未训练 |

MS-TCN 最佳在第 62 轮；后续训练损失继续下降，但末 30 轮验证窗口未超过 61–70 轮窗口，呈平台与波动迹象。当前没有依据要求 MS-TCN 因 SpecScalpel 的 300 轮预算而机械地追加至 300 轮，也不能证明它已达到理论极限。

ProTAS 41–50 轮平均分相较 31–40 轮仅提高约 0.46，最佳在第 46 轮。证据不能保证延长训练有收益，但也不足以判定 50 轮已经充分；补至既有 MS-TCN 的 100 轮预算是合理受控检查，保持学习率和其他设置不变。

## 4. 现有错误分析对方向的约束

已有 50 轮最佳权重验证对照显示：固定图 ProTAS 碎片更少，但动作漏检更多、前景准确率较低；漏检增加出现在所有时长组。其大量错误位于动作边界邻域外，因此不能直接将研究目标限定为“修边界几帧”，也不应优先增加全局平滑或更长确认延迟。

不过，该对照使用 MS-TCN 第 50 轮和 ProTAS 第 46 轮，不包括 MS-TCN 第 62 轮。上述现象是选题线索，不是充分训练后仍成立的最终结论。

暂定方法目标可表述为：**在保持因果输出的条件下，改善动作类别证据与前景召回，同时抑制片段碎裂。** 是否进一步采用边界协同、历史记忆或其他方法，应由更新后的错误形态及针对性文献核查决定；本次不认定某个模块具有创新性。

## 5. 下一步执行顺序

1. 等当前 SpecScalpel 训练完成，保留其原定协议，不用测试成绩调整在线方向。
2. 固定图 ProTAS 从第 50 轮续训至第 100 轮，维持同一训练/验证划分、模型、固定图、学习率和选模规则。本次只做了 `--check-resume`，已通过模型源码哈希、数据顺序、配置、checkpoint 与优化器兼容检查，没有创建训练进程。
3. 用 MS-TCN 100 轮与 ProTAS 100 轮各自验证选出的 best 重做验证集错误对照，比较漏检、碎片、混淆和边界偏移。
4. 在更新后的主要错误上检索改进机制、确定消融，之后再开展方法实验。SpecScalpel 为离线参照，保留其未来信息与文本先验差异，不把三者当成完全同条件排名。

### ProTAS 100 轮续训命令（当前 GPU 任务结束后执行）

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/train_small.py --model protas --run protas_pku_fixedgraph_e100_resume_seed17 --resume-from protas_pku_fixedgraph_dev_seed17 --split configs/dest_compat.json --epochs 100 --train-limit 0 --val-limit 0 --clip 0 --seed 17 --device cuda --stages 4 --layers 10 --width 64 --lr 0.0005 --normalization none --graph-reference official --task-graph fixed
```

此命令总目标为 100 轮，即补 51–100 轮，写新目录，不覆盖原 50 轮实验。恢复模型与 Adam 状态，不保证恢复所有随机状态。

## 证据与复查

- `convergence.json`：逐模型最佳记录、十轮窗口汇总、日志哈希与划分信息。
- `scripts/review_online_convergence.py`：纯日志复查脚本，无 torch 推理或训练。
- `reports/protas_fixed_graph/validation_comparison/REPORT.md`：历史 50 轮验证错误对照。
- 原运行：`mstcn_pku_dev_seed17`、`mstcn_pku_dev_e100_resume_seed17`、`protas_pku_fixedgraph_dev_seed17`。

复查命令：

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts/review_online_convergence.py
```

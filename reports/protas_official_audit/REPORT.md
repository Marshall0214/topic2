# ProTAS 官方一致性审计

2026-10-12。官方 https://github.com/Yuhan-Shen/ProTAS 。固定上游commit见sources/manifest.json。本轮没有修改适配模型或启动训练。

## 源码核对

实际下载官方main分支README.md、main.py、model.py、batch_gen.py、eval.py、utils/write_graph.py、utils/write_progress_values.py，与本地sources/protas/upstream逐字节比较：7/7完全一致，且与此前固定manifest的SHA256一致。详见source_checks.json。本轮下载发生在沙箱网络限制后经允许的网络读取，无上游文件覆盖。

## 实现核对

| 项目 | 判定 |
| --- | --- |
| MultiStageModel、GRU进度分支、融合、因果卷积 | 直接调用官方model.py，结构一致；输入维度与类别数为本地骨架配置 |
| CE、0.15截断平滑、进度MSE、0.1图损失 | batch1完整序列无padding时与官方公式一致 |
| cummax进度、图惩罚、eta=0.01解码 | 一致 |
| 进度标签 | 保留官方逐段从1/L到1、背景为0的定义 |
| 任务图统计公式 | 去背景后合并连续类别、计数及按列归一化与官方一致 |
| 任务图统计数据范围 | 官方工具扫描groundTruth目录；本地仅选定train池，防止验证/测试标签进入图 |
| 初始参考矩阵 | 本地build_model将官方共享存储的Parameter替换为detach().clone() buffer，实质改变正则与优化轨迹 |
| 训练预算 | 官方CLI默认50轮、batch1、Adam lr0.0005、4阶段10层64维；本地相同 |
| 输入及数据集 | 官方main默认2048维视频特征；本地300维骨架、52类槽位、PKU划分，不是原视频实验 |
| 随机过程 | 官方seed1538574472/Python shuffle；本地seed17/NumPy permutation且每轮重设torch种子，不是相同训练轨迹 |
| 权重选择 | 官方入口使用指定num_epochs对应权重；本地按独立val组合分数选best |
| 评价 | 官方eval.py同时报告含背景Acc和前景Acc-bg，含背景Acc与本地定义一致；官方Edit先去背景再合并重复类别，本地canonical切段去背景后不再次合并。Edit存在实质口径差异，不能直接与官方数字比较，也不能宣称精确复现DPTM表7 |

## 数值验收

verify_protas_official_parity.py使用相同随机初始化、同一合成batch1完整序列、相同dropout随机数，对照官方损失公式与本地training_loss。前向与图修正最大差0，loss差0，梯度最大差2.24e-8。此时参考图尚未发生漂移，不能外推为整个训练过程等价。

随后给两个模型的matrix_pre都加0.5：官方参考参数与可学习参数共享存储，参考正则仍为0；本地参考保持独立，正则为0.25。这证实本地改动真实影响目标函数，并非仅保存形式不同。详见parity_checks.json。

## 对当前故障的解释与修正结论

图参数无非负约束、图损失存在负向下降方向、只对pre设置参考正则等结构来自当前官方公开实现，不能说是本地凭空写错图公式。但目前发生退化的是本地骨架适配实验，不足以证明官方原视频实验必然失败。骨架类别、背景、动作顺序、训练更新次数与本地正则改动均会影响训练。

当前名称必须保留为“骨架输入适配 + 参考矩阵独立副本修复”，不能简称严格官方版。先前建议立即冻结图参数训练过早：如需建立官方一致对照，应先提供独立的official_exact_reference模式，恢复共享参考存储行为，保持其他骨架实验条件不变，与现有fixed-reference-copy版本做受控比较。旧权重不改、不直接切模式续训。固定图实验另列消融，不冒充官方learnable_graph设置。

## 重现数值检查

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' -B .\scripts\verify_protas_official_parity.py
```

此检查只创建合成图fixture，不训练、不评测真实测试集，不要求重新下载数据。

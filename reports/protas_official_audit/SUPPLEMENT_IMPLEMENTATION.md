# 官方任务图模式与统一补充评价

2026-10-12。现有验证选模仍使用 foreground_canonical 的 (Edit+F1@50)/2，未修改选模公式或已有best/last。没有启动新的完整训练。

## 重要纠正：CPU别名不能外推到CUDA

官方构造器的matrix_pre和matrix_pre_original在CPU共享存储；但是官方训练器会执行model.to(device)。在本机PyTorch2.8 CUDA环境，两个不同Parameter对象分别迁移，存储自动分离。CUDA下给pre加0.5，官方参考正则为0.25，而非0。因此此前仅凭CPU检查推断“本地独立副本改变了GPU正则目标”不成立。

重新对照官方构造模式和本地独立副本模式：相同初始化、合成输入及dropout随机种子下，GPU单步loss差0，梯度最大差2.61e-8；Adam单步状态最大差约1.81e-5，存在浮点累积差异，不能称逐位训练轨迹等价。此检查支持两者在本机CUDA下目标函数一致，不构成完整多轮等价证明。故暂不为参考矩阵这一点额外重跑50轮。

## 已实现

- baseline.build_model增加graph_reference='official'选项：直接返回官方模型，不做参考参数替换。默认independent_copy保留旧接口兼容。
- train_small.py增加--graph-reference；保存到config/checkpoint，续训模式不一致会拒绝。未修改分类、进度、图损失权重、图训练范围或选模规则。
- run_reference_baselines.py明确提供--graph-reference，默认official。
- evaluate_run.py按运行配置恢复对应模型，旧运行缺字段时用旧independent_copy模式。
- 新增OfficialMeter，统一输出protas_official：含背景Acc、前景准确率foreground_accuracy、官方Edit、F1及补充Avg。官方Edit先去背景再合并重复动作；F1仍按原时间轴切段、忽略背景。Avg仍是原五项平均，前景准确率不额外进入Avg。
- 原foreground_canonical、dest_legacy保留。--output-suffix用于另存评价，旧JSON不覆盖。

## 相同best权重的测试集补充评价

| 方法（50轮预算） | best轮次 | Acc | 前景准确率 | 官方Edit | F1@50 | 官方口径Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 因果MS-TCN | 50 | 58.29 | 44.13 | 45.54 | 25.44 | 41.64 |
| 骨架ProTAS | 25 | 52.05 | 27.47 | 42.83 | 21.90 | 38.61 |

234条test，236921帧，其中前景144325帧。只补充既定best的评价，没有在test选择权重/解码规则。补充文件为各run下test_best_evaluation_official_metrics_v1.json。原有canonical/legacy重算逐项与历史结果一致，详见supplement_summary.json。官方评价公式一致不代表骨架划分、输入及选模已精确复现DPTM。

## 验收

- 官方评测器随机序列对照：Edit与F1一致；合成[1,0,1]对[1,1,1]验证官方Edit=100及前景准确率=66.67。官方np.float旧接口仅在测试导入对象中兼容，上游文件未修改。
- 原verify_shared.py输入、指标、因果性检查通过。
- protas_official_behavior_smoke_v1：2条train/2条val、1轮GPU，官方模式训练及保存成功；重新加载best评价与训练记录一致。只证明软件流程，不代表准确率。
- 未因本轮补充指标改选checkpoint，也没有修改图学习约束或声称修复背景退化。

## 可选的新训练入口

若后续确需从官方构造模式启动独立实验，使用新目录：

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' -B .\scripts\run_reference_baselines.py --run protas_pku_official_dev_seed17 --model protas --device cuda --graph-reference official
```

此命令已具备入口支持，但本轮不建议仅为GPU下等价的参考正则行为重跑。不能直接拿旧checkpoint切模式续训。下一项研究应另行明确任务图退化的对照目的。

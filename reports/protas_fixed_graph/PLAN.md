# 固定初始任务图训练对照

日期2026-10-12。分支experiment/protas-fixed-graph。目的：判断学习任务图导致的参数漂移是否是现有骨架ProTAS后期退化的重要原因。此为适配审计/消融，非已成立创新。

## 冻结协议

相对已有protas_pku_dev_seed17，只改变图是否可学习。沿用300D骨架输入、698train/77val/234test、seed17、batch1、Adam lr0.0005、4阶段10层64维、完整序列、50轮预算。图仅从实际train生成；不合并trainval。保留GRU进度、融合、进度MSE、0.1图损失及eta0.01图修正。

新增--task-graph fixed调用官方learnable=False；两图参数requires_grad=False，仍参与图损失和解码。官方固定模式不添加学习图的参考正则；其固定时偏移为0，故没有额外正则梯度可移除。graph_reference=official直接使用上游构造行为；此前GPU参考副本与官方迁移后目标一致的核查见protas_official_audit报告，不把此作为第二项算法改动。

选模仍为原foreground_canonical (Edit+F1@50)/2，不使用补充官方Edit选模。不按test挑轮次或解码。新训练独立运行目录；不得将旧可学习图checkpoint切模式续训。resume入口已校验task_graph，默认learnable兼容旧配置。上游文件未改。

## 记录

每轮已有canonical/legacy及官方指标；新增validation_unadjusted_official记录最终分类输出未经任务图推理修正时的指标与背景比例，graph_diagnostics记录矩阵min/max与相对初始图最大漂移。修正后背景比例仍在validation_diagnostics。原first_stage_raw指第一阶段，不能当最终网络raw。

这些指标为诊断，不改变选模。固定图漂移应始终0；检查后期F1走势及修正前后差距。若消除背景崩塌但不提高精度，应报告“稳定性改善但无精度收益”，不能将两者混为一谈。

## 验收已完成

verify_fixed_graph.py在CUDA验证：图参数冻结、Adam后与初始图逐值相同、网络参数更新、图损失对分类和进度输入梯度非零、未来帧扰动和前缀因果检查通过。verify_shared.py原检查通过。

protas_fixed_graph_smoke_v1用2条train/2条val完成1轮GPU训练，保存best/last并重新加载评价；只验证流程，不能作为准确率证据。本轮未启动完整训练、未评价真实test。

## 完整训练（项目目录下PowerShell一行）

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' -B .\scripts\run_reference_baselines.py --run protas_pku_fixedgraph_dev_seed17 --model protas --device cuda --graph-reference official --task-graph fixed
```

默认50轮，698train/77val，新目录已通过dry-run配置检查。输出runs/protas_pku_fixedgraph_dev_seed17/protas/{best.pt,last.pt,history.json}和config.json、源码快照、训练图。历史结果不覆盖。完成后先查看验证曲线与图漂移，再按原best规则做一次test评价。

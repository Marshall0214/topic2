# 参考 DPTM 的骨架在线分割基线实验

日期：2026-10-12。当前已完成 PKU 实验配置、完整训练启动入口、统一评价与软件验收。正式完整训练由用户执行，尚未取得新实验结果。

## 证据与可复现边界

DPTM 原文：`D:\AQA\online_skeleton_action_segmentation\research_audit_ddcn\DPTM.pdf`。

- 第 8 页 Implementation Details：MS-TCN/ASFormer 主干、默认组件因果化、64 维、Adam、lr=0.0005、batch=1；DPTM 自身两阶段各 50 轮。
- 第 13–14 页骨架实验和表 7：LARA、PKU-MMD，引用 [4,16] 的标准协议，在 MS-TCN 中加入 1×1 卷积适配输入。
- 第 8 页指标：Acc、Edit、F1@10/25/50 及五项算术平均 Avg。
- ProTAS 官方仓库提供因果模型和训练；其 CLI 默认 4 阶段、10 层、64 通道、lr=0.0005、50 轮。
- DeST `train.py:100-103`：参数搜索使用 training；正式训练可切换 trainval。dataset.py 的 trainval 合并发布 train/val CSV。

来源：

- https://github.com/zczheng777/DPTM （本轮仍只有 README/LICENSE）
- https://github.com/Yuhan-Shen/ProTAS
- https://github.com/lyhisme/DeST

这是**参考 DPTM 设计的本地基线实验**，不是 DPTM 官方代码复现。论文未公开骨架的确切输入维度、坐标预处理、文件清单、背景排除和选模代码，也未确认上述一般训练细节是否对骨架另有调整。不能把未说明内容补成“论文明确要求”。

## 与此前小规模筛查的区别

| 项目 | 此前探索 | 当前正式参考协议 |
| --- | --- | --- |
| 训练 | 24/128 条子集 | 完整发布 train：698 条；val：77 条独立保留 |
| 评价 | 8 条验证，反复开发 | 每轮评价 77 条 val，训练完成后评价 234 条 test |
| 选择权重 | 组合验证分数选 best | 按 val 的前景 (Edit+F1@50)/2 选 best，不依据 test 挑轮次 |
| 方法 | MS-TCN/局部 ProTAS 试跑 | 同协议因果 MS-TCN、骨架 ProTAS |
| 输入 | 原值及标准化对照 | 固定原值，不继续归一化调参 |
| 指标 | Acc/Edit/F1 | 两套背景口径的 Acc/Edit/F1/Avg |

当前使用 DeST 发布的 698/77/234 划分，不合并 train/val。DPTM 的确切文件划分尚未核实。train/val 存在同采集跨视角重叠风险，需披露；test 按受试者分离。trainval 仅留待方法与超参数冻结后再讨论，当前入口禁用。测试序列历史上曾用于官方权重复现/探索，不能宣称完全未接触的全新盲测。

## 冻结配置

机器可读配置：`configs/dptm_reference_pku.json`；发布划分：`configs/dest_compat.json`。

- 数据唯一入口保持原路径；骨架完整连续序列，无片段拼接或真实边界裁剪。
- 输入固定为 300 维当前根相对坐标+后向位移，逐人处理缺失。这是本地透明选择，不宣称 DPTM 采用了这一具体输入。DeST 上游位移含下一帧，在线设置不能照搬。
- 首阶段已有 `Conv1d(300,64,1)` 承担输入维度适配，不另外叠加未经证实的 GCN/空间模块。论文说 1×1 适配不能推出其所有其他处理细节。
- 两模型 4 阶段、每阶段 10 层、64 通道、52 类槽位，严格因果；batch=1；Adam lr=0.0005，weight_decay=0；无学习率调度。
- 分类损失为每阶段 CE+0.15 截断时序平滑，阶段求和。ProTAS 额外保留进度损失、图损失及官方推理流程，图只来自实际训练池。
- 保留已披露的 ProTAS 任务图参考参数独立克隆修复，非作者源码逐位复现。训练图、源文件快照、配置和摘要均保存。
- 两基线各 50 轮，是冻结的本地训练预算并与 ProTAS 默认对齐。DPTM 自身的教师/学生两阶段训练不适用于 MS-TCN/ProTAS，不能机械改为 100 轮。

## 评价与报告规则

在 50 轮预算内，按完整 77 条验证集的前景 (Edit+F1@50)/2 选择 best（并列保留较早轮次），再在完整 234 条 test 上推理，先不加确认、平滑或多数投票。在线可见范围为当前及历史；当前前向支持整段因果等价计算，但不声称已经实现持久缓存或测得边缘实时性。

并列保存 `foreground_canonical` 与 `dest_legacy`，不根据哪套数值更高选择口径。Avg=(Acc+Edit+F1@10+F1@25+F1@50)/5。前者明确排除 ID0 前景片段，后者兼容上游 NONE 与端点约定。DPTM 的具体背景和 Edit 实现仍未知，原文数字单独标成 reported，不据当前差值宣称性能超越。

首轮 seed17 用于判断是否建立有效完整数据基线；之后若配置不再改变，再补 seed29/43，报告各次结果与均值/波动。若正式结果仍低，应先审查复现差异，不继续用小子集上的改进包装创新。

## 已完成软件验收

- `verify_shared.py`：输入、背景/端点样例、DeST 指标兼容、非零任务图下因果/前缀/状态隔离均通过。
- 新 Avg 计算已接入两套指标。
- `reference_trainval_smoke_v1`：使用 4 条已看过的开发序列作 1 轮 GPU trainval 软件检查，两模型均能保存 last，且不生成虚假的验证 best。
- 评价入口在人工小型 fixture 的 2 条开发序列上执行成功。fixture 的 test 字段仅为接口测试，不是正式 PKU test，结果不作为性能证据。
- 当前配置已改为 698 训练、77 验证、234 测试；trainval 的旧烟雾测试仅是历史软件验收，不代表当前实验协议。尚未启动长训练或本轮正式 test 评价。

## 完整运行命令

在 PowerShell 执行以下完整流程，依次训练两模型，再评价二者按验证集选出的 best：

```powershell
$py = 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe'
Set-Location 'D:\AQA\online_skeleton_segmentation_baselines'
& $py -B .\scripts\verify_shared.py
if ($LASTEXITCODE -ne 0) { throw '基线检查失败，请停止。' }
& $py -B .\scripts\run_reference_baselines.py --run dptm_reference_pku_dev_seed17 --model both --device cuda
if ($LASTEXITCODE -ne 0) { throw '训练失败，请保留报错和运行目录。' }
& $py -B .\scripts\evaluate_run.py --run dptm_reference_pku_dev_seed17 --model both --checkpoint best --split test --device cuda
if ($LASTEXITCODE -ne 0) { throw '评价失败，请保留报错。' }
```

也可运行已保存的 `RUN_DPTM_REFERENCE.ps1`。不需要重新安装环境或下载数据。同名 run 不覆盖，重跑需修改运行名；此入口目前没有断点续训，意外中断先保留 last 与日志，不手动删除已有结果。

输出：`runs/dptm_reference_pku_dev_seed17/{mstcn,protas}/best.pt`、各 history.json、summary.json、reference_protocol.json、`test_best_evaluation.json`。每轮打印训练 loss 和验证指标，保存 best/last；234 条 test 不参与模型选择。

## 第二数据集

DPTM 同时报告 LARA。当前本地 official_data/dataset 仅有 PKU-subject，因此本次先落实 PKU；LARA 的数据入口、关节数和特征形状需独立核验，不能用 25 关节 PKU 代码冒充 LARA 已复现。论文最终多数据集验证仍未完成。

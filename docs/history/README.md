# 因果 MS-TCN 与骨架版 ProTAS：统一小规模训练

> 最新正式基线方案：参考 DPTM 的设计，见 [DPTM_REFERENCE_PROTOCOL.md](DPTM_REFERENCE_PROTOCOL.md)。698 条训练、77 条验证、234 条测试，按验证集选择 best；trainval 暂不采用；下文小规模命令保留为探索实验记录。

日期：2026-10-12。实现是统一协议下的本地基线适配，不声称精确复现 DPTM 表 7。原始数据和上游源码保持原样。

## 环境与目录

已用 `C:\Users\admin\anaconda3\envs\dest_gpu\python.exe` 实测：Python 3.10、PyTorch 2.8.0+cu126、CUDA，现有环境已具备 numpy、pandas、tqdm。无需重新安装 PyTorch，也不需要下载骨架数据。当前实现 batch=1，不依赖多进程 DataLoader。

工作目录：`D:\AQA\online_skeleton_segmentation_baselines`。

数据入口：`D:\AQA\online_skeleton_action_segmentation\official_data\dataset\PKU-subject`。

## 直接执行：先检查，再依次训练两种模型

在 PowerShell 粘贴整个代码块：

```powershell
$py = 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe'
Set-Location 'D:\AQA\online_skeleton_segmentation_baselines'
& $py -B .\scripts\verify_shared.py
if ($LASTEXITCODE -ne 0) { throw '共享输入、评测或因果性检查失败，请停止训练。' }
& $py -B .\scripts\train_small.py --model both --run small24_val8_e10_seed17 --epochs 10 --train-limit 24 --val-limit 8 --clip 0 --seed 17 --device cuda
if ($LASTEXITCODE -ne 0) { throw '训练失败，请保留报错和该运行目录。' }
```

`--model both` 顺序训练因果 MS-TCN，再训练 ProTAS，不会同时占用两份训练显存。两者使用同一组选定训练/验证序列，每种模型训练 10 轮；每轮验证并按前景 `(Edit+F1@50)/2` 选 best。它们结构不同，因此不宣称两者初始参数相同。

`--clip 0` 表示训练和验证均使用完整连续序列，不按真实动作边界裁剪。`--train-limit 24` 是 24 条序列，不是 24 帧。数据子集按固定种子选择，实际类别覆盖写入 config；这个实验仍是小规模筛查，不能当完整数据性能。

同名运行目录存在时会报错，避免覆盖。重跑请修改 `--run` 名称；当前小规模入口不提供断点续训。

如果想分开运行，使用以下两条命令替代上面的 both 命令；无需同时执行两种方式：

```powershell
& $py -B .\scripts\train_small.py --model mstcn --run mstcn_small24_val8_e10_seed17 --epochs 10 --train-limit 24 --val-limit 8 --clip 0 --seed 17 --device cuda
& $py -B .\scripts\train_small.py --model protas --run protas_small24_val8_e10_seed17 --epochs 10 --train-limit 24 --val-limit 8 --clip 0 --seed 17 --device cuda
```

两条命令选择相同子集。GPU 不可用时会明确报错，不会偷偷降到 CPU。运行命令前无需 `conda activate`，因为已指定环境内 Python 的绝对路径。

## 输出及查看

both 命令生成：

```text
runs/small24_val8_e10_seed17/
  config.json                    # 实际序列、类别覆盖、超参数、版本及摘要
  train_graph.pkl                # 仅选定训练序列标签生成的图
  source_snapshot/               # 训练入口、共享实现、上游模型和划分快照
  summary.json
  mstcn/best.pt
  mstcn/last.pt
  mstcn/history.json
  protas/best.pt
  protas/last.pt
  protas/history.json
```

控制台每轮打印 loss、验证集 Acc/Edit/F1@50 与耗时。完整指标和两套背景口径保存到 history.json。默认显示前景片段口径，不等同 DPTM 论文报告数值。本入口不自动读取测试序列或计算测试成绩。

补充诊断：日志现在同时显示 `pred_bg` 与 `first_stage_F1`；全部预测背景时会明确 WARNING。若 Acc 恰好等于真实背景比例且前景 F1 为 0，不能将 loss 下降当作已经学会动作。首个 10 轮运行确实出现该情况，已完成权重及单序列拟合检查，详见 `reports/ACC_STALL_DIAGNOSIS.md`；原短训练命令仍保留作为历史筛查配置。

查看汇总：

```powershell
Get-Content .\runs\small24_val8_e10_seed17\summary.json
```

## 统一部分与模型差异

共享输入：`(T,150)` → 逐人当前根相对 XYZ 150 维 + 后向 XYZ 位移 150 维 → `(300,T)`，单位保留发布数据原值。每人独立检查缺失，首次出现或相邻帧缺失时位移清零。无未来插值、前向差分或整段归一化。

共享主干配置：4 阶段、每阶段 10 层、64 通道、52 类。共享 Adam lr=0.0005、batch=1、各阶段 CE + 0.15 截断时序平滑损失；按原风格将各阶段损失求和。序列级训练无 padding，因此没有填充值参与损失。

- 因果 MS-TCN：沿用 MS-TCN 的阶段与 softmax 精修结构，复用固定版本 ProTAS 的因果残差卷积，没有进度 GRU、融合头或任务图。阶段权重独立随机初始化；它是本地因果实现，非 DPTM 官方 MS-TCN 训练代码。
- 骨架版 ProTAS：复用官方因果模型，保留每阶段单向进度 GRU、分类/进度融合、可学习任务图。另加权重 1 的进度 MSE 和权重 0.1 的图损失，图正则沿用官方 0.01、推理 eta=0.01。图仅由此次实际选定训练序列构建；进度真值仅用于训练，推理只使用模型预测。
- 本地明确修复：官方 graph original 参数与可学习矩阵共享存储，现改为独立克隆的固定 buffer，使参考矩阵不随训练漂移。上游文件未修改。该修复不是论文创新。

模型参数量：因果 MS-TCN 703440；适配 ProTAS 844048。该比较是方法比较，不是完全等参数消融。

默认采用 `configs/recording_holdout_seed17.json`，避免同次采集不同视角跨 train/val。若需原 DeST 兼容划分，显式添加 `--split configs/dest_compat.json` 并使用新的运行名，两套协议分别报告。

## 统一评价器及检查

- `foreground_canonical`：Acc 包含全部有效帧；Edit 与片段 F1 排除 ID 0；半开区间、末端为 T；Edit 先提取真实片段，保留 A-background-A 中两个 A；F1 汇总跨序列计数。前景双方均为空时 Edit=100，全为空的 F1=0；一方为空按漏检/误报处理。此约定写入检查样例。
- `dest_legacy`：计入 NONE，保留原评分器末端 T-1 约定；已与原 DeST 评分器在固定随机分段样例上验证数值一致，容差 1e-5。对原实现会出错的空集/零分母做安全处理，因此不声称所有病理输入都逐位一致。

`scripts/verify_shared.py` 检查逐人缺失、首次位移、输入未来扰动、单帧和纯背景计分、漏检/误报、进度目标、评分兼容、图参考值独立性、两个模型未来扰动与前缀一致性。详细结果见 `reports/shared_verification.json`。

已完成 GPU smoke：`runs/smoke_shared_v1`，两种模型各 2 条训练、2 条验证、2 轮，均成功保存 best/last，loss 下降。该样本仅有 13 个标签值，验证前景 F1 为 0，只证明运行链路，不能评估方法好坏。

当前前向属于全序列因果计算；逐前缀核验是重放前缀，不是持久缓存流式实现。本次不据此报告边缘逐帧时延。小规模实验下一步看学习趋势、前景预测及两者稳定性，再决定完整训练和错误分析。

## 128/8 输入标准化单变量对照

已新增 `--normalization train-standard`，默认 `none` 保持旧行为。统计只来自本次实际选择的训练序列，逐通道排除缺失人体和无效相邻位移；验证使用固定参数，不对验证序列拟合。标准差下限 1e-3，无训练观测通道采用 mean=0、scale=1，缺失值仍为零，不加裁剪。

```powershell
$py = 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe'
Set-Location 'D:\AQA\online_skeleton_segmentation_baselines'
& $py -B .\scripts\verify_normalization.py
if ($LASTEXITCODE -ne 0) { throw '标准化检查失败。' }
& $py -B .\scripts\train_small.py --model mstcn --run mstcn128_val8_e50_norm_seed17 --epochs 50 --train-limit 128 --val-limit 8 --clip 0 --seed 17 --device cuda --normalization train-standard
if ($LASTEXITCODE -ne 0) { throw '训练失败，请保留报错。' }
```

与 `mstcn128_val8_e50_seed17` 比较，除标准化外保持数据、初始化种子、训练预算、模型与评分相同。统计保存在新 run 的 normalization.npz，配置记录来源训练列表和 SHA256，后续推理必须加载同一统计。`diagnose_run.py --run mstcn128_val8_e50_norm_seed17` 已支持按配置自动加载统计。尚不保证改善，也不将标准化视为论文创新。

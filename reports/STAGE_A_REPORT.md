# 阶段 A 验收报告

日期：2026-10-12。已完成数据审计、固定版本源码下载、训练/推理静态检查和无训练合成输入因果性检查。没有新训练、没有新测试集性能评价、没有修改上游代码或已有结果。

## 来源核验

| 方法 | 固定版本 | 实际状态 | 下一步用途 |
| --- | --- | --- | --- |
| ProTAS | 0f04ee6b3c4370ce22f9230bca42e9aabe47b428 | 有训练循环、优化器、权重保存、整段因果/逐前缀推理 | 主要在线代码起点；需做骨架输入和训练清单适配 |
| MS-TCN | 33ed91c0c7576650a2367efc602553af4c5295b1 | 官方原版离线；Trainer 保留 Python 2 语法 | 基础主干参考，不能直接当在线实现 |
| DPTM | 6945f9dba176869f17c132333e95a24d30d1bbc7 | 仅 README、LICENSE | 论文参考，尚不可直接训练 |

源码分别存放 `sources/<name>/upstream`，提交号、文件 SHA256 与许可证元数据见 `sources/manifest.json`。ProTAS 仓库本轮未发现独立许可证文件，不将其推断为 MIT；MS-TCN 许可证需阅读原文，GitHub 元数据标记 Other；DPTM 为 MIT。这里只做研究核验与本地复现准备，不替作者重新授权。

官方网站：

- https://github.com/Yuhan-Shen/ProTAS
- https://github.com/yabufarha/ms-tcn
- https://github.com/zczheng777/DPTM

## 代码级发现

ProTAS `model.py` 中有分类 CE、截断时序平滑、进度回归、任务图损失以及完整训练保存流程；因果配置采用裁除右侧输出的卷积与单向 GRU，图推理使用累计最大进度。

需要明确处理的适配点：

- `main.py` 默认视频输入维度 2048，数据目录和映射格式针对原视频数据，不能直接传 PKU 路径。
- `model.py` 的 Trainer 在关闭图模块时仍访问 graph_learner；无图控制需单独修正调用分支。
- `utils/write_graph.py` 遍历整个标签目录，没有训练清单过滤；必须防止验证/测试标签进入图。
- `eval.py` 使用过时 np.float；其背景名称、Edit 去背景合并顺序及末端约定与 DeST 不完全相同。
- 原生 predict_online 重算完整前缀，没有持久卷积/GRU 缓存，不能用整段推理速度宣称流式每帧开销。

## 合成输入检查

使用未训练官方 ProTAS 模型：4 阶段、每阶段 10 层、64 通道；输入维度 300、52 类，33 个时间步；任务图为合成零矩阵。CPU、PyTorch 2.8.0+cu126。检查阈值 1e-5。

- 改变第 17 步及之后的输入：前 17 步分数最大变化 0。
- 0、1、7、16、32 五个位置逐前缀最后一帧与整段因果结果比较：最大差异 8.94e-8。
- 相同输入重复执行：最大差异 0。
- 离线 MS-TCN 对照检测到未来扰动影响，变化约 1.20e-5。该对照只执行未修改的模型定义，跳过含 Python 2 print 的 Trainer。

结论：该配置的因果执行与前缀一致性检查通过，可以继续验证骨架适配。这里只验证实现性质，不证明准确率、学到有效进度、真实任务图性能或缓存推理效率。

环境补装 tqdm 4.70.1 与 colorama 0.4.6；其余使用既有 dest_gpu 环境。未安装新训练框架。

## 数据与协议风险

详见 `../protocol.md` 和 `data_audit.json`。主要发现：1009 条数据完整；原 train/val 的 77 次基础采集跨视角重叠；背景 NONE 不会被默认 background 排除；mapping 52 类但实际仅 42 个标签值；DeST 前向位移不能用于零等待在线输入。

新增原切分兼容清单和按采集分组的 699/76/234 清单，不覆盖上游 CSV。第二套开发切分确保验证集覆盖全部实际类别，测试不参与选择。原文分数与本地结果尚不可直接比较。

## 验收与后续动作

阶段 A 通过本地统一协议小规模验证的准入；尚未通过 DPTM 精确复现准入。

下一步进入阶段 B：复用已核验 ProTAS 因果代码，准备纯因果 MS-TCN 基础对照及 ProTAS 骨架适配的统一数据入口和评测器，优先用分组验证清单。先测输入因果性、训练梯度、少量样本学习与在线输出，再安排完整训练。输入适配和因果控制本身不计作创新。

复核命令（PowerShell）：

```powershell
$py = 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe'
Set-Location 'D:\AQA\online_skeleton_segmentation_baselines'
& $py -B .\scripts\audit_protocol.py
& $py -B .\scripts\prepare_split_candidates.py
& $py -B .\scripts\verify_source_causality.py
```

`fetch_sources.py` 会按执行时仓库最新提交重新下载，禁止为重现本报告无意更新来源；复核时直接使用当前固定版本与摘要。

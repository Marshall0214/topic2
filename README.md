# 第二研究点：连续骨架序列在线动作时序分割

本目录是后续开发入口。由 D:\AQA\online_skeleton_segmentation_baselines 完整复制，原目录保持不变。

## 当前入口

- [训练、续训、测试命令](docs/TRAIN_TEST_COMMANDS.md)
- [代码与实验管理](CODE_MANAGEMENT.md)
- [实验索引](docs/EXPERIMENT_INDEX.json)
- [现有在线基线结果](reports/online_convergence_review/RESULTS_E100.md)
- [CurvSeg 审查](reports/curvseg_audit/AUDIT.md)
- [迁移验收](reports/workspace_migration/RESULTS.md)

## 目录职责

src/ 是本地模型适配与共享输入评价；scripts/ 是执行入口；configs/ 是实验配置和固定划分；sources/ 是固定上游代码及资产；reports/ 是可提交的分析；runs/ 是完整运行快照、权重和日志；docs/ 是使用说明，docs/history/ 保留整理前文档。

## 环境与数据

Python：C:\Users\admin\anaconda3\envs\dest_gpu\python.exe。
数据继续只读引用 D:\AQA\online_skeleton_action_segmentation\official_data\dataset\PKU-subject，未复制或迁移数据。环境不重装。
模型与输入源码保持原字节，避免破坏旧运行的续训校验；当前 DATA 入口仍在 src/baseline.py。历史运行内的绝对路径为来源记录，不做批量替换。评价入口将旧项目内划分路径映射到本目录，并继续核对原 SHA256。

新终端先执行：Set-Location 'D:\topic2'。命令中的 epochs 表示目标总轮数，不能将历史 50 轮命令当成当前必须重跑的任务。

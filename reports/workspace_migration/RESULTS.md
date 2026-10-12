# topic2 整理验收

2026-10-12。目标：独立整理第二研究点，不修改或删除其他目录。

## 完成

- 从 D:\AQA\online_skeleton_segmentation_baselines 复制到 D:\topic2，保留 .git、全部分支、源代码、固定上游文件、配置、报告和完整 runs/。
- 复制时 1,105 个非缓存项目文件逐一 SHA256 一致；Git 对象 fsck 通过，原分支引用全部保留。新副本分支 chore/topic2-workspace。
- 建立首页、更新代码管理规范，整理 docs/ 与 docs/history/，生成 31 组运行索引。旧根文档提供导航，训练命令改用新工作目录。
- 不修改 src/ 模型代码，历史 runs/、上游原快照不变。保留历史绝对路径作为来源证据；评价器仅解析项目内划分文件到新目录，并核验原哈希。
- 旧 DeST 评价参考代码复制进 sources/legacy_dest_reference，带来源哈希，消除 verify_shared 对旧研究代码目录的依赖。
- 原始数据继续引用原目录，未移动或复制；两个历史审计工具仍有明确的外部资料入口，属于只读来源，不宣称项目已完全脱离 D:\AQA。

## 验证范围与结果

| 项目 | 结果 |
| --- | --- |
| 三个模型 best 权重严格加载 | 全部通过，epoch 分别为 62、75、119 |
| 验证序列前 96 帧推理 | 全部通过，类别输出均为 [1,52,96]，数值有限 |
| 输入、共享指标、旧评价口径、因果前缀与状态隔离检查 | verify_shared.py 全部通过 |
| 固定图 ProTAS 续训检查 | 100→101 的 --check-resume 通过，未建立训练运行或更新权重 |
| SpecScalpel 训练状态恢复 | last=158；模型、Adam、历史及 RNG 字段可加载；未执行优化器更新，不声称逐步续训等价 |
| MS-TCN 直接续训 | 被原有源码哈希保护拦截，见下 |
| 原项目保护 | 整理结束复核 1,105 文件均未变化，原 Git 工作区干净；副本 runs/ 与复制时仍逐文件一致 |

本轮没有完整验证集重评、测试集评价或训练。短序列检查仅证明迁移后的加载和推理链路可用。

## MS-TCN 历史续训注意

runs/mstcn_pku_dev_e100_resume_seed17/source_snapshot/baseline.py 是训练时版本，迁移前的当前 src/baseline.py 后续已增加 ProTAS 模式与补充评价。两者文件哈希不同，旧续训入口按整文件哈希校验，所以直接续训失败。新旧项目当前 src/ 逐字节相同，故该问题不是迁移引入。

保留原保护，没有改历史配置或白名单放行。今后确需续训该 MS-TCN 时，应在独立版本/工作区恢复训练时实现并验证；当前三个模型仍可加载进行评价。现有基线无需因为搬目录重新训练。

## 文件

copy_manifest.json：复制时来源、大小与哈希（不含 .git 元数据和缓存）。
original_preservation.json：原项目和副本运行产物的结束核验。
verification.json：模型和训练状态检查，checkpoint_and_inference_passed=true，all_resume_checks_passed=false。
../shared_verification.json：共享检查。

## 后续入口

工作目录 D:\topic2；环境保持 dest_gpu。先阅读 README.md、CODE_MANAGEMENT.md 和 docs/TRAIN_TEST_COMMANDS.md。从整理提交建立新实验分支，旧项目留作原始备份，不在两处同时开发。runs/ 不进入 Git，仍需单独备份权重。

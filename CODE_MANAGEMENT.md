# 代码和实验管理规则

## 当前状态

唯一的新开发入口为 D:\topic2。当前整理分支 chore/topic2-workspace 从 ecdb4ca 建立，继承全部旧分支和提交；main 是旧基准快照，不代表已包含后续审查与实验。未配置远程，不自动合并旧 main。

## 保留边界

- D:\AQA 以及其他位置不修改、不删除。本目录 runs/ 为副本，不是原目录的链接。
- 模型 src/、上游 sources/ 原快照、历史 runs/ 内配置、权重、日志、源码快照保持不变。新增本地评价参考单独注明来源。
- runs/ 受 Git 忽略，Git 不能代替权重备份。本次 copy_manifest.json 记录复制时的逐文件哈希。
- docs/history/ 和 reports/ 的旧路径、旧分支说明属于历史记录；以本文件与首页作为当前导航。

## 新实验

从本次整理完成的提交建立 experiment/<方法名> 分支。修改前检查 git status；不要在多个目录同时开发。上游文件不原地修改，适配放 src/ 或 scripts/。不重写基线运行配置，不覆盖旧运行目录。

运行命名建议：方法_数据协议_关键变体_seed编号_v编号。配置记录划分、输入、选模规则和全部参数；每次运行保存命令、Git commit、源码快照和哈希。仅用验证集选方案，测试集在方案冻结后使用。

## 当前正式参照

- 因果 MS-TCN：runs/mstcn_pku_dev_e100_resume_seed17
- 固定图骨架 ProTAS：runs/protas_pku_fixedgraph_e100_resume_seed17
- 离线 SpecScalpel：runs/specscalpel_full_seed17

其余运行仍保留，属于探索或检查，不删除，不与上述结果混写。实验索引 docs/EXPERIMENT_INDEX.json 可定位全部有 config.json 的运行；权威配置在对应 runs/ 中。

## 命令与文档

训练和评价命令见 docs/TRAIN_TEST_COMMANDS.md，模型专属参数不能混用。历史协议文档可能记载早期训练预算，后续实际运行配置优先。不要因整理目录而重新训练。

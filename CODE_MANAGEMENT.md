# 代码与实验管理

本项目目录原无 Git，于空间编码筛查开始前初始化本地仓库。

- `main`：已有因果 MS-TCN/ProTAS 适配、统一输入评测与续训代码的基准快照，提交 `f322ff4`。这是历史代码快照，不表示所有历史方案已经验证有效。
- `experiment/spatial-encoding-pilot`：当前空间编码实验分支。训练实现提交 `8765a52`。
- 没有配置远程或推送。上游源码以已下载的固定文件管理，原仓库来源与 commit 保留于 `sources/manifest.json`，不修改上游文件。

## 数据与结果

继续使用原 PKU-subject 数据入口，不下载/复制数据。`runs/`、权重、NumPy 二进制、缓存图文件均忽略，不进入代码版本库。旧训练结果不删除、不覆盖；新实验分别使用带编码器后缀的独立目录。小规模汇总数字、验证协议和解释写入 `reports/spatial_pilot/` 并提交 Git。

每个新运行保存配置、实际序列清单、输入和源码哈希、源码快照、best/last、history/summary。启动器额外保存 `experiment_provenance.json`，包含启动时 Git commit 和完整命令。实验记录以运行快照为准，后续分析脚本可能来自更晚的分析提交。Git 不备份被忽略的模型权重；权重继续保存在本机原目录。

## 兼容性与边界

原 `src/baseline.py` 不改，新增编码器放在 `src/spatial_encoders.py`。训练入口新增 `--spatial-encoder linear|mlp|gcn`，默认 `linear`，评测依据 run 配置选择结构。现有线性模型 checkpoint 兼容；续训检查也已通过。实验编码器续训会额外校验对应源码哈希。

本轮只训练 MS-TCN 三种输入编码变体，不训练 ProTAS，不评价测试集。三组完整训练预算与验证选模规则相同。未来修改编码器时先创建新配置/运行目录，禁止原地混用 checkpoint。

## 常用命令

```powershell
Set-Location 'D:\AQA\online_skeleton_segmentation_baselines'
git status --short
git log --oneline -5
git diff main...experiment/spatial-encoding-pilot -- src scripts configs
```

保留当前实验分支；在筛查完成并检查结果前，不自动合并回 main。

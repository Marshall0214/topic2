# CurvSeg 论文与公开代码审查

审查日期：2026-10-12。目标：判断是否补充现有骨架在线动作时序分割基线。没有训练、修改上游代码或使用测试集预测。

## 结论

CurvSeg 是高度相关的骨架时序分割工作，尤其直接覆盖“分类与边界相互促进”。但当前固定版本的公开代码存在启动、依赖完整性、损失配置和梯度传播问题，不能作为已经可以复现的基线。原模型具有未来依赖，不能直接加入在线方法的同组比较。

建议保留现有因果 MS-TCN、骨架适配 ProTAS、离线 SpecScalpel 三个实验基线，将 CurvSeg 列为优先对照的相关工作与候选离线基线。先取得完整、与论文一致的代码或明确记录修复，再决定是否投入训练。此审查不能证明论文实验错误，也不能证明所有版本都有这些问题。

## 来源与固定版本

- [ICLR 2026 正式论文页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/78b44e74db9f9b31256135ca1049179e-Abstract-Conference.html)
- [正式论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/78b44e74db9f9b31256135ca1049179e-Paper-Conference.pdf)
- [官方仓库固定版本](https://github.com/kong-johnny/CurvSeg/tree/3b42e7fd74f2e2351562f7ffa4e3f5f0e7a93e83)
- 本地 sources/curvseg/manifest.json 固定 131 个文本源码与配置文件的 SHA256；审查脚本验证全部一致。
- reports/curvseg_audit/remote_tree.json 保存远程完整目录索引。未下载数据、权重或二进制文件。

论文提出用分类特征轨迹曲率指导边界定位，并用定位结果反向约束分类特征；双专家模块提供任务相关特征。因此仅提出“双分支 + 分类边界交互”不足以区分我们的贡献。论文任务是骨架时序分割，但不因此自动符合在线输出协议。

## 已核验事项

| 项目 | 结果与证据 | 含义 |
| --- | --- | --- |
| 数据划分 | PKU-subject train1/val1/test1 分别为 698/77/234；样本名集合与 configs/dest_compat.json 全部相同 | 可复用当前划分；不等于预处理、损失、评价器已经一致 |
| 参数解析 | train.py:51 将 type 写成 typr；独立执行真实 get_arguments 函数触发 TypeError | 修复前训练入口无法正常解析参数 |
| 必需源码 | 远程索引缺少 prompt/text_prompt.py、prompt/tools.py、libs/loss_fn/circle.py、libs/perturb.py；训练入口/helper 引用它们 | 完整依赖源码尚未公开在这些路径；不能将本地未下载与远端缺失混淆 |
| 已存在的模块 | libs/loss_fn/rdrop.py、libs/models/SP.py、libs/swap.py 在远程及本地均存在 | 不应误报这些文件缺失 |
| 协同损失权重 | pku_sub.sh 只传 --dataset；train.py 的 --cross 默认 0，实际用 args.cross 构造 alpha=cross、beta=cross/2；配置 YAML 的 cross:4.0 未用于这一处赋值 | 即便修好启动问题，原启动命令仍未启用该协同损失；有限输入探针验证损失为 0 |
| 曲率梯度 | curve.py:81 的 torch.tensor(embs, device=device) 切断计算图；helper.py:295 等使用这一计算结果 | 单独通过此曲率约束，梯度不能回到输入分类特征；不是说整个分类分支完全没有梯度 |
| 双向损失实际传播 | 非零 alpha=4、beta=2 时，边界 logits 梯度范数 0.0187502，分类特征梯度为 None | cross.py 虽写双向 detach，一侧输入已被更早断开；共享骨干仍可能经边界分支间接更新，不能混同为论文所述直接曲率反向约束 |
| 推理未来依赖 | 配置所用模型的 Linear_Attention 对全部 keys/values 求和，无因果遮罩；STI 对全序列自适应池化 | 原版属于离线比较；不能简单改成左填充就宣称完成因果化 |
| 注意力数值检查 | 独立执行原注意力类，前 40 帧输入不变，只修改第 40 帧之后；前 40 帧输出最大变化 0.350685 | 证实该模块未来依赖；不是完整模型端到端评测 |
| 曲率时间依赖 | 双侧差分、全序列归一化；修改后半段后前半段先验最大变化 0.520974 | 若仅用于训练，使用未来监督本身不排除在线学生；需要将训练约束与推理依赖分开判断 |

梯度与数值检查输入为随机张量，CPU、PyTorch 2.8.0+cu126、seed=17。用于核查实现性质，不用于推断真实数据精度。

## 接入前仍需完成的工作

1. 核实完整作者实现，解决缺失依赖，确认启动参数和曲率反向传播与论文预期一致。任何本地修复放在适配代码中，上游快照保持原样。
2. 核对骨架预处理、人物选择、通道定义、采样尺度、背景标签、边界生成与后处理。划分相同只解决了样本集合问题。
3. 保持现有验证集选模规则与统一评价器；如另报作者口径，单独列出，不能直接混用论文表格数字。
4. 原版若复现，应列为离线方法。若研究在线适配，应重新定义状态缓存、可用历史和确认延迟，并对全流程做未来扰动检查。

当前不安排 CurvSeg 完整训练，也不把“修复公开代码”或“将离线模型因果化”直接认定为研究创新。我们的方法选题需要进一步围绕在线条件下的实际错误，提出区别于现有分类—边界协同机制、且有对照实验支撑的改进。

## 复核命令

在 D:\AQA\online_skeleton_segmentation_baselines 中执行单行 PowerShell：

```powershell
& 'C:\Users\admin\anaconda3\envs\dest_gpu\python.exe' scripts\audit_curvseg.py
```

输出：reports/curvseg_audit/probes.json。此脚本只读取上游源码，执行独立的解析器、损失及注意力探针并写出审查结果，不训练模型、不下载资源。

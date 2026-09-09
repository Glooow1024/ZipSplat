# Scene-token 50k 后续训练操作记录

日期：2026-09-09。用户接受上一轮关于扩数据、重设学习率、有限扩大 DA3 适配的建议。本轮执行独立的续训阶段，保留旧运行和结果。

## 固定训练口径

| 项目 | 配置 |
|---|---|
| 父运行 | `/root/multiview_compare/experiments/scene_token/main_20260908_v2` |
| 起点 | `checkpoints/step_050000.pt`，严格恢复全部模型参数 |
| 新运行 | `/root/multiview_compare/experiments/scene_token/continue_20260909_v1` |
| 新数据 | `/data/datasets/3dvision/ZipSplat-scene-token/expanded_20260909_v1` |
| 目标规模 | 1024 train + 原32 validation；最终有效数量以 summary.json 为准 |
| 更新预算 | 追加100000次全局更新，累计50000→150000；每步8场景，每场景4target |
| 输入 | 固定6个context，252px，沿用原采样/归一化；target可能与context重叠 |
| 模型 | S256，继承原std0.02训练出的参数，不重新初始化；distinct1D不变 |
| 跨视角attention | 保留patch+scene完整交互，没有改为scene-only |
| DA3解冻 | 从原30–39扩大为18–39；0–17与其他原冻结模块保持冻结 |
| 损失 | 与50k运行相同，包括LPIPS0.05、max_scale正则阈值0.1；没有新增loss |

第18层起解冻，使第19/29/39层三路特征前都有可适配block；并非解冻整个DA3，也不是已验证最优层数。数据、解冻、学习率同时改变，本轮是下一阶段训练，不作为单变量因果实验。

学习率按参数名精确分组，前1000个新增更新从峰值的10%线性升至峰值，此后余弦衰减至峰值的10%，不会越过终点自动反弹。

| 参数组 | 峰值LR | 起点/终点LR |
|---|---:|---:|
| scene token和颜色query | 1e-4 | 1e-5 |
| 新解冻DA3 blocks18–29 | 1e-6 | 1e-7 |
| 原已解冻DA3 blocks30–39 | 3e-6 | 3e-7 |
| 原下游 | 3e-5 | 3e-6 |

AdamW旧参数的一阶/二阶矩与step按名字迁移；新解冻参数没有历史状态。scaler及八rank RNG从父checkpoint恢复，新阶段scheduler重新开始。同一新运行中断恢复时则恢复本阶段完整scheduler，不重启warmup。严格固定父checkpoint SHA256、父配置hash和本轮源码快照。

## 数据与启动验收

- 沿用原冻结的训练/验证/官方测试划分。确定性顺序选取源数据审计通过的1024个训练场景。相机检查包含有限值、齐次矩阵、旋转正交/行列式、轨迹尺度和极端跳变，以及图像文件存在性。
- 本次源数据筛查排除了3个极端轨迹场景，包括先前已发现的 `48aaed5a...`，另两个为 `0888cdee...` 和 `98341043...`。这是保守异常筛查，不是所有标签语义正确的证明。
- 已校验并复用287个tar（255旧训练+32验证），优先hardlink，无额外重复占用；新增769个场景使用原DA3 teacher及转换口径。新manifest共360145帧，预计总tar约27–30GiB，实际以summary计数为准。旧tar/原始数据不改。
- 转换后逐tar重新验证SHA256，并检查深度有限/正值/上下界顺序、绝对范围和相对轨迹尺度。未通过训练场景不进入新index，原tar留存；训练有效数低于请求的97%或验证有变化/缺失则停止。转换worker异常退出也阻止训练。
- 8卡数据转换与后台训练管理独立于SSH运行。数据ready后自动执行CPU优化器/真实采样验收，再启动8卡训练。
- 第一次模型更新前，重放原50k固定train_probe/V2/V6全部场景，检查相同context/target和数值差异；阈值为逐场景PSNR<0.01dB、LPIPS/total<0.001，不称逐bit一致。不通过则停止。
- CPU已测试参数分组LR、跨阶段moments保留、新参数空状态、学习率下限与序列化恢复下一更新完全一致（小模型），以及变更验证target被拒绝。它不等同真实大模型GPU恢复已通过。
- 首50个新增更新（累计50050）进行评估、完整checkpoint与关键参数回读；第一个和每100步检查scene、颜色query、head以及DA3 blocks18/29/39的有限非零梯度，优化器更新必须实际发生。之后每2000步评估并保存，保留最近3份、里程碑及V6最佳PSNR/LPIPS checkpoint。

## 网页和日志

- 新运行：<http://127.0.0.1:18766/>，默认V6验证，增加同场景V6训练参考；训练参考不属于held-out。初期显示的50k图像从父运行导入，实际恢复评估会覆盖该参考条目。
- 旧0–50k结果：<http://127.0.0.1:18767/>，旧日志、checkpoint和图片保留。
- `status.json`、`metrics.jsonl`、`rank*.jsonl`、`evaluations.json`、`stage_transition.json`、`continuity_check.json`、`first_checkpoint_check.json`、`checkpoints.json`均位于新运行目录。
- 数据进度/审计在新数据目录的 `progress.json`、`source_audit.json`、`depth_audit.json`、`summary.json`；总转换日志 `/root/multiview_compare/logs/scene_token_expand_20260909_v1.log`。
- 全流程遇错会显示failed并保留日志，不自动重试或静默裁剪异常loss。STOP在训练安全边界保存后退出；数据等待期间STOP取消自动启动，数据转换进程独立继续。

停止本轮：

```bash
ssh vllm1 'touch /root/multiview_compare/experiments/scene_token/continue_20260909_v1/STOP'
```

只在用户决定继续且已有本轮checkpoint时恢复：

```bash
ssh vllm1 'cd /root/zipsplat/ZipSplat && rm /root/multiview_compare/experiments/scene_token/continue_20260909_v1/STOP && .venv/bin/python tools/scene_token/launch_training.py --run /root/multiview_compare/experiments/scene_token/continue_20260909_v1 --resume'
```

若尚无本轮checkpoint，不能使用上述resume；需先诊断失败，按父checkpoint初始化路径重启管理器。旧main_v2已完成，不应在旧目录改总步数恢复。

本轮进展以PROJECT_CONTEXT最新记录和远端status为准，部署不等于后续100k训练已完成。未提交/推送。

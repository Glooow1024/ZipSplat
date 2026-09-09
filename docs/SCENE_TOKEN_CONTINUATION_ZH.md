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

### 2026-09-09：修复数据准备进度读取竞态并恢复管理程序

- continue_20260909_v1 的训练管理程序在读取 progress.json 时发生 FileNotFoundError，此前单独 exists() 成功，文件在检查时已恢复可读。数据转换进程仍正常运行；故障发生时尚未启动新训练，无新增优化器更新，父50k checkpoint未受影响。
- run_training.py改为直接读取，针对短暂不存在/不完整JSON/ESTALE、EIO、超时进行有限重试；暂时缺失时保留上次进度及其原始时间，连续5分钟没有有效心跳仍停止。最终summary也使用同一路径读取；持续损坏、权限错误及passed=false仍阻止启动。
- tests/test_training_preparation_poll.py新增6项回归测试全部通过，覆盖文件短暂消失、旧心跳保留与过期、初次无文件超时、JSON重试/持续损坏、IO重试/权限失败和成功或失败的最终summary。
- 已确认没有训练进程/metrics，再将旧status、pipeline日志、provenance和旧脚本归档到新运行recovery_history/poll_fix_1788919868；仅修正run_training.py及其对应源码快照/hash，provenance.pretraining_repairs明确记录前后hash和原因。159个冻结源码hash重新通过，没有放开校验或修改训练配置。
- 已重启管理程序PID29969，viewer和8个转换worker未重启。检查时progress882/1056、0转换失败；网页回到preparing_data。数据完成后仍须通过审计、恢复重放和首50步验收；不把管理器恢复称为新训练已开始。

### 2026-09-09：约92700步梯度守卫报错与恢复诊断

- rank2在92700步触发train_main中“每卡全部监测梯度必须>0”的assert，其他7rank已记录该步有限非零局部梯度；全局日志到92699左右，最近完整checkpoint92000。原assert未输出触发参数/数值，不能事后确定是局部零还是缺失hook，更不能直接认定模型整体梯度失效。
- 只读单样本探针：用92000权重重放92700/rank2确定性输入（scene4ab881b3...），loss0.07253、activated约.885，六个监测梯度均有限非零。这是更早权重的探针，不是原失败状态精确重现；报告gradient_sample_probe.json留运行目录。
- 修复检查口径：保留原局部hook，允许有限局部零值；在optimizer.step前记录DDP汇总、AMP unscale和clip之后的实际梯度。仍要求hook完整、全局实际梯度有限非零与优化器计数增加，错误保存gradient_failure_rank*.json，局部零/缺失情况写gradient_observations_rank*.jsonl；每rank日志新增optimizer_gradients，不改参数更新算法。
- tests/test_distributed_gradient_monitor.py两项通过，其中真实2进程CPU/Gloo测试复现local0/2而汇总均1、两rank共同更新权重；缺失、NaN/Inf、全局零检查仍报错。没有因局部零而跳过batch或裁剪loss。
- 原状态/日志/源码/provenance归档到recovery_history/gradient_fix_1788951429；仅更新train_main及对应159文件清单中的hash，provenance.repairs记录前后版本。管理器从完整92000恢复并将后续旧日志归档recovery_history/1788951452，未保留的约700步重算。新管理PID5192、torchrun5199，模型/optimizer/scaler/RNG/scheduler按既有恢复路径加载，未重启warmup。
- 已验证恢复首步8卡实际梯度逐值相同，前10步LR与旧日志逐值相同。重放训练数值不是bitwise一致（前10步最大loss差6.4e-5、PSNR差.00765、LPIPS差.000732），不宣称精确重现原失败状态。目前已过92330且未有新错误，须继续观察原92700位置。
- 同时只读确认原50k续训初始化continuity_check为passed、原三集合PSNR/LPIPS/total差均0；50050 checkpoint关键参数回读passed、optimizer567条。它们是先前启动验收，不是本次92000恢复的新checkpoint验收。

- 恢复后已正常越过92700，最终核验status running/92740、无error_rank，159项源码hash一致。同一92700/rank2样本本次六个局部及汇总梯度均有限非零，loss .070228、activated .87636；未重现原断言，也没有局部零观察记录。不能声称已确定原故障是哪个参数的零梯度。本轮已修正不合理的逐卡非零判据并补齐诊断，但原始异常的具体数值原因未被精确复现。
- gradient_fix_verification.json记录越过原失败步及完整该rank指标；模型结构/数据/loss/LR配置不变，训练继续后台运行，后续每2000步checkpoint策略保持。

### 2026-09-09：确认并修复AMP梯度监控范数的FP32溢出

- 新故障发生在93100/rank2，gradient_failure_rank2.json明确记录Gaussian head局部norm=inf，其他局部norm正常，实际用于优化器的该参数norm=.54744且其他汇总norm正常。局部hook逐元素isfinite断言先通过，随后g.float().norm()产生inf：这次根因是监控的FP32平方累加溢出，不是已发现梯度张量含Inf或优化器更新非有限。上一轮仅修正逐卡非零规则，没有解决此数值问题；不能继续归因于局部零梯度。
- 92000 checkpoint的scaler.scale=4611686018427387904（2^62）、growth_interval2000。先计算scaled梯度FP32范数再除scale会在未缩放范数约4以上时溢出。改为torch.linalg.vector_norm(...,dtype=float64)累加后再除scale，同时对实际优化器范数用相同稳健函数；仍拦截真实非有限元素/非法scale、缺失hook和全局零梯度。未改变AMP scaler、模型、loss、LR、裁剪或优化器更新。新增日志amp_scale_before/after及失败记录中的scale。
- 新增test_amp_gradient_norm.py四项通过，包含用实际checkpoint scale复现旧误报、真实CPU GradScaler保持正确更新、零/极小/极大有限值、真实NaN/Inf/非法scale拒绝。与原双进程DDP两项合计6项通过。
- 额外在服务器H100上对3000000元素、同2^62缩放做数值回归：所有元素finite，旧norm溢出，新norm5.196152467873376，与先FP64解除缩放参考完全相同。报告gpu_amp_norm_regression.json，非额外模型训练实验。
- 再次确认无活动训练进程后，将故障日志、gradient_failure、旧源码/provenance归档recovery_history/amp_norm_fix_1788962667，更新train_main及对应source快照和hash，在provenance.repairs保留完整原因。旧失败日志另由恢复管理器归档，不删除历史。
- 从最近完整92000 checkpoint恢复，重算未保存的约1100步；新管理PID46864、torchrun46911，首步AMP scale与checkpoint一致、各监测量有限。已正常到92070、无error_rank；这次尚未重放到93100，修复依据是已复现的数值溢出及CPU/H100回归，不声称已越过本次原故障步。

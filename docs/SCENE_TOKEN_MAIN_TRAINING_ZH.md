# Scene token 分阶段主训练与监控

日期：2026-09-08。运行名：`main_20260908_v2`。

用户选择扩充数据后直接按阶段训练，不再追加固定拟合、初始化尺度或 max_scale 的预实验。本轮是中等规模 DL3DV 主训练，尚不是全量数据或论文复现；代码保留在 `dev/scene_token`，未提交或推送。

## 访问与运行目录

- 实时网页：[http://127.0.0.1:18766/](http://127.0.0.1:18766/)，每 15 秒读取服务器最新记录。
- 服务器训练目录：`/root/multiview_compare/experiments/scene_token/main_20260908_v2`。
- 数据目录：`/data/datasets/3dvision/ZipSplat-scene-token/main_20260908_v1`。
- 代码：`vllm1:/root/zipsplat/ZipSplat/tools/scene_token/`；环境：该仓库自己的 `.venv`。
- Windows 结果目录只保留配置、工程验收和启动摘要；持续变化的完整日志、图像、checkpoint 以服务器为准。网页通过 SSH 隧道直接读取服务器，不依赖手工下载。

服务器上的数据转换、训练管理进程和网页服务均脱离 SSH 会话运行。关闭聊天、浏览器或本地电脑不会停止服务器训练。本地 SSH 隧道断开后网页暂时不可访问，在 PowerShell 中重新运行：

```powershell
& 'F:\Development\multiview\ZipSplat\tools\scene_token\open_training_monitor.ps1'
```

随后打开上述网址。新网页端口 18766，原 18765 服务及历史结果保留。

## 数据和监督口径

已转换 **256 个训练场景、32 个验证场景**，排除 1 个原始相机异常的训练场景后，当前实际使用 **255 个训练场景、32 个验证场景**。沿用原始冻结的 scene-level 划分和确定性排序，原官方测试场景保留，训练与验证没有场景交集。本次训练不使用验证场景梯度，也不加入 RE10K。

8 个 GPU 转换 worker 使用已有 DA3-Giant teacher、GT 相机参数和原始 960p 图像，复用已验证的坐标变换、252px 预处理、lossless WebP 与 log-u8 伪深度流程。teacher 的多帧分组只在同一个场景内，因此验证场景不进入训练标签生成。保留逐场景来源、teacher 配置、输入帧信息、tar 哈希和失败日志。新数据单独存储，不覆盖旧 pilot 或此前 held-out 隔离实验。

训练每 GPU 每步 1 个场景、4 张监督图，全局 batch=8 场景，无梯度累积。所有 rank 遍历同一轮场景随机排列的不同位置；跨轮重新排列，记录每次实际 context/target 和 pose guard 重采样次数。有限 CPU 缓存为每 rank 8 个原始 tar、512 个解码视图，避免随场景数持续增长。

context 和 target 使用仓库原 `BoundedViewSampler`：target 在 context 时间跨度内分层随机采样，**允许部分 target 与 context 重合**，这是当前采样器本身的行为。context 顺序打乱，使参考相机变化；仍使用官方相对相机、深度归一化及 pose-jump guard。训练跨度随输入数增加，参数为最小跨度 8、`gap_multiplier=11`、最大相邻 context gap=22；本轮不同时扩展到论文的 24 个输入视角。

固定评估使用 32 个完整隔离的验证场景。每场景固定同一 18 帧跨度、8 张 target，V2 和 V6 输入嵌套且都与 target 不重合。另取 8 个训练场景作为训练参考，**这些 target 没有从训练及 teacher 中排除，不能叫 held-out 测试**。三组均记录全部相机索引。评估不做 target pose refinement。

## 50,000 步训练配置

从 released 完整 ZipSplat 权重重新初始化，scene token 正态初始化 std=0.02、S=256，使用 distinct 一维 slot RoPE。维持 252px、原 patch+scene 注意力结构和颜色 cross-attention；下游模块参与训练。每视图 256 个 scene token，不再通过 k-means 进一步减少它们。

| 全局优化步 | 输入视角 | 可训练范围 |
|---|---:|---|
| 1–2500 | 2 | scene token、颜色 query、原下游模块；旧 DA3 参数冻结 |
| 2501–4499 | 2 | 以上模块，并解冻 DA3 blocks 30–39 |
| 4500–6499 | 3 | 同上 |
| 6500–8499 | 4 | 同上 |
| 8500–10499 | 5 | 同上 |
| 10500–50000 | 6 | 同上 |

8×H100 80GB，BF16、activation checkpoint、DDP。每阶段有自己的 LR warmup 和 cosine：第一阶段 warmup 250 步，联合阶段 warmup 500 步，从峰值的 10% 开始，随后该阶段 cosine 衰减到零。峰值 LR：新 scene token/颜色 query `3e-4`，原下游 `3e-5`，解冻 DA3 `3e-6`；不按全局 batch 额外放大。AdamW、weight decay 0.05、betas=(0.9,0.95)、gradient clip=1。

阶段切换从 2500 步完整 checkpoint 加载模型，按参数名称迁移原有 AdamW moments，为新解冻参数创建新状态；LR scheduler 启动联合阶段的 warmup。同一阶段恢复则恢复原 scheduler，不重启 warmup。保存全部 8 个 rank 的 RNG、global step、样本数和 scaler。

RGB L1 权重 1、LPIPS VGG 0.05、位置损失 0.1、深度损失 0.01，以及原有 GS 正则；详见 `SCENE_TOKEN_LOSS_AND_TRAINING_ZH.md`。**本轮明确设置 `max_scale=0.1`（官方 YAML）；历史 pilot 实际为 1.0。** 其余阈值同样显式保存。由于还增加了数据规模、解冻、课程和 scheduler，本轮不能作为某一单独变化的因果对照。

固定 S 时，V2/V6 总 token 分别为 512/1536，GS 数为 16384/49152；V6 同时增加输入信息和输出预算，不能解释为同预算的纯视角增益。50k×8 共 400,000 次场景曝光，255 场景时平均每场景约 1568.6 次；不是每张图获得 50k 次监督。

## 自动记录与网页

| 文件或目录 | 内容与频率 |
|---|---|
| `metrics.jsonl` | 每步八卡均值：loss、RGB L1、位置/深度损失、PSNR、LPIPS、激活 GS 比例、LR、视角数、时间 |
| `rank0.jsonl` … `rank7.jsonl` | 每步各卡全部损失/健康指标、实际采样帧、重采样次数、峰值显存；首步及每 100 步记录关键参数梯度 |
| `status.json` / `dashboard.json` | 通常每 10 步及加载/评估/保存/退出时更新；网页每 15 秒刷新 |
| `evaluations.json` / `renders/step_*/` | 初始 0 步、50 步、每 1000 步、阶段末的固定三组评估；每次 72 组场景/视角数、共 576 张目标的 GT 和 render |
| `checkpoints/` / `checkpoints.json` | 首 50 步、每 1000 步、阶段末、安全暂停时，原子保存完整训练状态 |
| `effective_warmup.json` / `effective_joint.json` | 实际合并后的模型、head、训练配置、参数组和可训练参数量 |
| `config.json` / `provenance.json` / `source/` | 固定配置、released 权重 SHA256、Git 状态、源码快照及 SHA256 |
| `pipeline.log` / `warmup_*.log` / `joint_*.log` | 管理进程及各训练阶段的 stdout/stderr |
| `error_rank*.json` | 未捕获的训练异常；管理进程同时把错误写入网页 |

网页提供每 100 步训练 loss 均值、固定验证 V2/V6 PSNR 和 LPIPS 曲线、最近两次评估差值；可选择训练步、历史步、集合、场景、目标图并列看 GT/历史重建/当前重建。PSNR/LPIPS 为所选场景全部 8 张 target 的均值。输入视角数变化时训练样本难度改变，应优先根据固定验证曲线和图像判断进步。

checkpoint 保留最近 3 份、2500/10000/25000/50000 里程碑，以及验证 V2 的最佳 PSNR 和最佳 LPIPS；只有本次运行目录内不再满足保留规则的周期文件会被轮换。历史评估图和指标继续保留。保存前检查剩余磁盘至少 40GiB；首 50 步重新读取 checkpoint，核对关键 tensor、优化器条目及 8-rank RNG 结构。阶段切换/恢复前校验源码没有偏离快照，恢复时还校验配置哈希。

NaN、关键梯度缺失/非有限、优化器跳过更新、超过重采样限制或任一 rank 出错都会停止本次管理流程并保留日志。不会自动重试失败或自行判断收敛提前停止。网页超过 3 分钟没有活动状态更新时会提示检查进程；暂停、失败、完成状态不会误标为仍在正常训练。

## 暂停和恢复

在本地 PowerShell 发出安全暂停请求：

```powershell
ssh vllm1 'touch /root/multiview_compare/experiments/scene_token/main_20260908_v2/STOP'
```

等待网页显示 `paused`，训练会在当前 step/评估完成后的安全边界保存再停止，不应强杀正在写 checkpoint 的进程。数据准备阶段的 STOP 仅取消自动启动训练，不终止独立的数据转换。

已有训练 checkpoint 时恢复：

```powershell
ssh vllm1 'rm /root/multiview_compare/experiments/scene_token/main_20260908_v2/STOP'
ssh vllm1 'cd /root/zipsplat/ZipSplat && .venv/bin/python tools/scene_token/launch_training.py --run /root/multiview_compare/experiments/scene_token/main_20260908_v2 --resume'
```

如果异常退出而没有 STOP，直接执行第二条。恢复前先排查异常；管理脚本拒绝并行启动同一运行。若训练尚未开始、只有数据准备取消，则去除 STOP 后使用相同启动命令但省略 `--resume`。

崩溃可能留下最新 checkpoint 之后的日志。恢复脚本先将日志/评估备份到 `recovery_history/时间戳/`，只保留 checkpoint 所覆盖的记录继续追加，将超出步数的图像移入该备份目录，避免把未被恢复的参数轨迹展示成连续训练。

## 本轮验收与后续

已通过 CPU 工程检查：阶段切换保留旧 optimizer moments、新参数组无错误继承状态、序列化后的下一次 CPU 更新一致，以及真实数据 V2–V6 预处理和固定验证窗口。此项没有新 GPU 拟合或参数消融，也不等于已经验证未来联合阶段的完整性能或分布式逐位重放。

当前v2启动验收已通过，详见服务器运行目录的 `startup_check.json`：八卡首50步连续且全部损失有限并低于异常阈值；首50次global更新已覆盖全部255训练场景；0/50步的2304张PNG完整、252×252；7,371,983,283字节checkpoint重读通过，171个optimizer参数状态及8-rank RNG存在。核验时已继续至约100步，无本次异常记录。热身阶段单卡步中位约0.412–0.416秒、峰值9.72–9.74GiB，此速度不能外推联合训练V6。本次正常启动后独立运行，不等待50k完成，也不提前声称泛化已解决。

## 用户追问：2500 步解冻与 256 场景的依据

2500 是本轮工程起点，不是通过消融证明的最佳解冻时间。global batch=8 时共20,000次场景曝光，256场景平均约78.1次；仅表示初步适配新模块，不代表每张图充分训练。解冻限最后10层，DA3峰值LR3e-6为新参数的1/100。ZipSplat论文是单阶段联合微调DA3，其5% warmup是学习率热身，并非先冻结5%；GVC1D公开pretrain.sh给出约5k步的建议，并注明最好看loss收敛，不能脱离batch和任务直接套用。因此没有证据仅凭步数判定2500必然过早，冻结更久也不保证更好。[ZipSplat §4.1](https://arxiv.org/html/2606.05102v1#S4.SS1)、[GVC1D pretrain.sh](https://github.com/GVC1D/GVC1D/blob/main/pretrain.sh)

256仅占本地9820训练候选的2.61%，作为最终泛化训练数据偏少。本地下载清单10581、旧完整10485，进一步过滤13个候选并排除135个已有官方测试后得到10337，冻结为9820训练/517验证；官方test完整名单141个。本次只转换其中256/32，不把其余候选说成已经逐图转换合格。

ZipSplat使用DL3DV+RealEstate10K等比例混合，但原文未明确最终清洗后实际场景总数。DL3DV原始论文10510视频、RealEstate10K官方约80000片段（来自约10000原视频），这些是原始数据集统计，不能直接相加当作ZipSplat训练数，也不能用原始论文版本替换当前下载清单10581。GVC1D附录报告OpenVid-HD 433523视频、Vimeo-90K七帧序列以及约6000额外32帧序列；公开OpenVid训练清单核对为433523行。视频片段与独立三维场景不可等同。[DL3DV](https://arxiv.org/abs/2312.16256)、[RealEstate10K](https://google.github.io/realestate10k/)、[GVC1D Appendix A](https://arxiv.org/html/2603.15302v1#A1)

建议下一次数据扩充优先1024–2048场景，再向9820候选推进；相较只延长冻结时间，增加场景多样性更值得优先。此为后续建议，当前2500/256配置未因这次问答自动改变，也没有额外启动数据转换。

## 启动期间修正与数据排除

旧启动运行 `main_20260908_v1` 保留，不作为正式连续训练结果。首次 torchrun 命令在模型加载前因 `--run` 缩写解析冲突退出，添加参数分隔符后修复；第一次50步checkpoint已完整保存并重读通过，但索引中误把 `stat().st_size` 当作函数，修正后实际从50步恢复，最终安全暂停至203步checkpoint。日志、旧代码和修正来源在旧运行的 `startup_fixes/` 和 `recovery_history/` 中。

随后发现原始场景 `1K/48aaed5a44005bccd51d529ab90335b144fe5e7f3c8a22ba399f4ee3b3fb6728` 的385–394帧原始位姿平移范数达到约4.3e16，场景其他帧中位约3.52。异常位姿传播到teacher多帧深度对齐，深度上界达到5.97e17，造成旧运行41/81步巨大的有限深度loss。此前的有限值、旋转正交及图像有效率检查不足以发现这种原始数据异常；“288个转换成功”仅表示转换程序未报错，不等于288个全部适合训练。

已遍历288个场景全部98332帧的存储深度范围及相机平移，仅发现上述1例灾难性量级异常；其余场景最大存储深度上界约772。使用 `depth_audit.json` 和配置中的 `data.excluded_scenes` 显式排除该场景，不删除或覆盖原图/标签。实际训练变为255场景、验证仍32场景。审计是量级检查，不是全部伪深度几何正确性的证明。

当前有效运行 **`main_20260908_v2` 从released重新初始化**，不继承接触过异常监督的旧参数。S/std、loss系数、2500步阶段和50k预算不变；新增加规范化深度绝对值超过1e6或非有限时阻止输入，以及 **total loss 非有限或绝对值超过100时，在反向传播前停止**。这两个是异常熔断，不做loss截断、重加权或静默跳过。CPU检查已在新255/32配置通过，包含全部32验证窗口。

监控继续使用127.0.0.1:18766，服务已切换到v2；旧运行保持暂停和原记录。后续恢复请使用本文v2命令，勿恢复旧v1。

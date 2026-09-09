# S256 初步训练验证（2026-09-07）

**结论：训练更新与 checkpoint 恢复通过；高质量小样本拟合未通过。** 当前结果只证明损失可下降、参数可更新，不足以证明场景 token 能替代原 patch 表征。暂不扩大到全量数据或长训练。

## 配置与范围

- 权威运行：vllm1 `/root/multiview_compare/experiments/scene_token/initial_train_20260907_v3/`。
- 单张 H100，BF16，B=1，每次 2 个 context / 4 个 target、252×252、S=256、16,384 个 GS。关闭二次 k-means、GT 先验和 query ratio schedule。
- 复用完整 released ZipSplat checkpoint；已有 DA3 参数冻结。更新 scene tokens、颜色 query 投影、prepare/几何 CA/SA、颜色分支和高斯 head，约 1.973 亿参数。
- AdamW，betas=(0.9,0.95)，weight_decay=0.05，clip_grad=1。新参数共 589,952 个，最高 LR=3e-4；已有下游参数 LR=3e-5。前 10 步 LinearLR 从 0.1 倍升至 1 倍，之后保持。原骨干本轮没有微调。
- 原 DataLoader/官方 bounded sampler 为 8 个训练场景各生成一个固定 batch，随后缓存 CPU batch；前 100 步拟合第一个 batch，后 100 步轮流拟合八个 batch。训练采样允许重叠，不额外筛选有利帧。不是遍历 3,326 帧、不是全量随机训练。
- 固定单样本是书架场景 `7e6564c7...`，零基 context `[186,194]`，target `[187,189,190,193]`。这一样本恰好互斥，不代表所有训练样本强制互斥。
- 验证只使用先前固定的 2 个独立场景，6 context + 8 个互斥 target；直接用相对 GT 相机及 context-depth 归一化尺度渲染，不做 Sim(3)/逐视角优化或 TTO。验证指标与此前 pose-aligned 压缩预实验、论文正式 benchmark 不可直接比较。
- 测试每 50 步评估固定训练/验证数据，保存逐步各项损失、梯度和图像。所有正式测试场景保持未使用。

## 实测结果（v3）

| 检查 | 结果 |
|---|---|
| 完成更新 | 主训练轨迹 200 步，无跳过；另做 2 次恢复一致性诊断更新并回滚到保存状态 |
| 固定单样本 loss | 初始 0.57182 → 第100步 0.21234，下降 62.9% |
| 8个固定训练batch平均loss | 0.53393 → 0.24302，下降 54.5% |
| 两个验证场景平均loss | 0.62082 → 0.32412，下降 47.8% |
| 验证PSNR | 4.502 → 10.768 dB |
| 原版同预算验证PSNR | 14.988 dB，仍高约 4.22 dB |
| 梯度和参数 | 必查槽位/颜色投影/prepare/head梯度非零且有限；所有冻结参数前后SHA256一致 |
| 纯更新计时 | 第11–200步中位数0.210秒，平均0.220秒，p90约0.228秒 |
| 显存 | 恢复检查后正常训练/验证段peak allocated约9.75 GiB；含恢复检查峰值约12.55 GiB |
| checkpoint | 第100/200步各约7.37GB；模型、AdamW、LR scheduler、AMP scaler、计数及RNG状态保存 |
| 恢复一致性 | 相同batch/RNG下，新建模型/优化器从磁盘恢复，下一步loss差0；全部可训练参数最大绝对差3.83e-6（阈值1e-5），AMP scaler状态完全相同 |
| 回归测试 | 9项通过，包含新增梯度缩放器增长边界的恢复检查 |

计时使用已缓存的 CPU batch，包含 forward/render/loss/backward/AdamW，但不含磁盘 DataLoader、验证、checkpoint、模型初始化；不能直接外推 V=24、全骨干微调或8卡DDP吞吐。AMP梯度钩子记录的范数已除以当步scale；不能把缩放后的大数误判为梯度爆炸。

原版参考复用 released checkpoint，压缩保留率256/324，V6时49,152 GS，与新模型V6/S256完全同预算，使用相同图像、相机、分辨率和损失配置。该两场景参考最初在v2后独立生成，已验证v2/v3模型配置、数据配置及完整帧清单一致后复用；不是从v2训练权重生成，也不是挑选最好结果。

## 图像与表征诊断

训练与验证图像都仍以大块颜色和模糊轮廓为主，未恢复可用的局部几何和细节。PSNR提升并不等于学会高质量重建；固定训练样本也模糊，说明问题不只来自验证场景未见过或训练/验证视角数量不同。

在同一书架训练样本、同K=512上检查**几何融合后的 token**：

| 指标 | Scene 第100步 | Scene 第200步 | 原版同预算 |
|---|---:|---:|---:|
| token平均两两余弦相似度 | 0.9555 | 0.9391 | 0.4185 |
| 去均值特征的熵有效秩 | 2.68 | 2.49 | 17.44 |
| GS尺度p90（归一化场景） | 0.3110 | 0.2854 | 0.0327 |
| 固定样本PSNR（独立重载测量） | 13.50 | 13.17 | 22.26 |

有效秩由特征去均值后的奇异值平方归一化，再计算 `exp(entropy)`，不是矩阵的精确代数秩。相似度/低有效秩和较大的高斯尺度与模糊图像共同提示表征趋同、几何分工不足。**这些是诊断信号，尚不能定位为DA3槽位、几何融合、颜色分支或冻结策略中的某个单一原因。** 只有200步、8个固定batch，不足以据此否定整个研究思路，也不能断言S=256容量不够。

图像：`training_curves.png`、`validation_before_after.png`、`fixed_training_comparison.png`；细节见`token_diagnostic.json`和`analysis.json`。本地副本在`ZipSplat/results/scene_token/initial_train_20260907_v3/`，不下载大checkpoint。

## 本轮修复与历史运行

1. 测试工具调用原`Trainer.train_step`前须像正式epoch循环一样重置StepTimer；v1因未初始化计时器在第一次更新前退出，**0次更新**，保留失败记录。
2. v2完成200步并验证一步数值恢复，但复核发现原Trainer未保存/恢复AMP GradScaler。BF16配置在本仓库同样启用该scaler，缺少增长计数会影响长期恢复。因此给`Trainer.save_checkpoint`补入scaler状态，给`load_checkpoint(load_state=True)`增加可选恢复，兼容旧checkpoint；新增CPU AMP回归测试跨越growth_interval验证数值/scale一致。
3. v3在相同固定配置上重跑200步并包含scaler状态，是本轮权威结果。v2保留作排错记录，不用其checkpoint继续训练。v2/v3短程训练结果有差异；CUDA算子未设置严格确定性，不能把同seed当成完整轨迹逐位复现保证，也未按质量择优选版本。

工具：`train_initial_test.py`、`report_initial_test.py`、`evaluate_initial_baseline.py`、`diagnose_initial_tokens.py`。仅修改本轮工具、Trainer的scaler持久化及文档；无新分支、无提交/推送、未改原始数据。

## 下一步优先级

P3仍未通过“小样本高质量拟合”验收，不直接启动大规模微调：

1. 分层检查19/29/39的DA3场景特征与几何融合后的槽位差异，定位趋同出现在哪一步；同时检查颜色注意力和GS位置/尺度覆盖。
2. 在同一固定样本、同一相机/监督上做更充分的拟合诊断，并观察图像细节，不能只追逐loss下降。
3. 单因素比较“保持原DA3冻结”与“解冻后部层”的适配效果；根据证据再考虑位置/槽位初始化或辅助监督。一次只改变一项，不先凭低质量结果增加S。
4. 小样本细节重建通过后，再扩大场景与视角数量、验证DDP/保存恢复/真实数据加载吞吐，最后进入全量训练。

上述后续尚未执行。本次200步检查完成并退出，没有后台长训练。

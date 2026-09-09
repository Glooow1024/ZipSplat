# Scene token 训练数据 pilot（2026-09-07）

本轮按用户授权处理少量真实数据，先验证数据转换、伪深度、DataLoader 和完整损失反传；不启动长训练。沿用 `dev/scene_token` 单工作区，S=256。

## 1. 论文、官方实现和本项目的选择

来源：[ZipSplat 论文](https://arxiv.org/pdf/2606.05102) §4.1、§4.2、附录 A/B；[官方 bounded sampler](https://github.com/cvg/ZipSplat/blob/main/splatfactory/datasets/view_sampler/bounded.py)；[官方 DL3DV split](https://github.com/cvg/ZipSplat/blob/main/splatfactory/datasets/scripts/dl3dv/split.json)。

- 论文训练使用等比例 RE10K / DL3DV，252×252，2–24 个 context，每步渲染 4 个 target；使用 DA3-Giant 配合 GT pose 产生伪深度。
- 论文没有说明训练 context/target 必须互斥。官方 bounded sampler 在轨迹范围内分段随机采样 4 个 target，没有排除 context 索引，允许部分输入视角重建监督。根据本轮用户要求，**保留官方训练采样器，不增加训练互斥限制**。此前文档中“训练也严格持出”的建议被此决定取代。不要把可能重叠说成原实现必然有错。
- 论文测试使用 8 个持出 target；DL3DV 用 6/12/24 个 context、FPS 和 50/100/150 的帧间隔设置。论文报告 140 个 DL3DV / 1,600 个 RE10K 测试场景；这是 benchmark 规模，不等于下载目录的完整 test 索引规模。
- 没有找到论文单独定义验证集的说明。采用项目自定义的**场景级约 95%/5% train/validation**；同一场景所有帧只能属于一个划分。最终官方测试留到训练方案确定后，不能用其选 S、学习率或 checkpoint。已有用于选参的 9 个压缩预实验场景属于开发数据。
- 本轮仅 DL3DV 小样本，尚未构成论文等比例混合训练。后续正式训练前需明确模型初始化、分组 LR、优化器与混合数据规模。本轮真实模型损失系数沿用 ZipSplat 的 RGB=1、LPIPS=0.05、Chamfer=0.1、depth=0.01，与论文一致；不能误用独立 GaussianHead 类的默认 Chamfer=0.05 来描述 ZipSplat 实际配置。

## 2. 冻结的场景划分

输入是上一轮 `environment_20260907/training_candidates_252.txt` 的 10,337 个候选场景；已经排除已知缺帧、帧数不足、首帧短边不足，以及官方 test。该清单不是全量逐图解码证明。

按 SHA256(`scene-token-split-256:` + `bucket/hash`) 排序，前 ceil(10,337×5%)=517 个用于验证，其余 9,820 个用于训练。另以 `scene-token-pilot-256:` 哈希排序从各自分区选出 8 个训练、2 个验证场景。清单固定，后续扩大数据时不重新混洗 train/val。

官方 `split.json` 有 141 个 test ID，本机已有 135 个；全部 141 个均保留，不把缺失的 6 个当作可以用于训练。与当前候选的场景 ID 交叉为零。Evaluation55 保持独立，本轮不重划分 RE10K。

pilot 的固定验证为每个场景前 150 个注册帧内的 6 个 context + 8 个互斥 target，使用 `eval_sampler` 重放固定索引。它是低成本的本项目验证窗口，**不是论文 FPS benchmark 的复现**。最终论文对比另用官方 benchmark 索引；不能把 pilot 指标与论文或原 NoPrior 可视化指标直接合并。

## 3. 数据与 teacher 处理口径

派生目录（vllm1）：`/data/datasets/3dvision/ZipSplat-scene-token/pilot_20260907/`。

源目录：`/data/datasets/3dvision/DL3DV-10K/`，保留每个入选场景全部注册帧及原顺序，不只抽几张。原图不删除、不改写。最终每场景一个 tar，有独立 `train/index.json` / `validation/index.json`。

1. 读取 transforms.json 和 images_4 的实际图像，逐帧解码，检查尺寸和有限相机/位姿值。与官方 converter 相同，以实际图宽缩放 K，裁剪到主点并恢复原图尺寸；OpenGL c2w 翻转相机 Y/Z 轴，转换为 OpenCV c2w。
2. teacher 读取上述原尺寸 RGB，`process_res=504`、`upper_bound_resize`，显式输入 K 与 w2c，`align_to_input_ext_scale=True`，不运行 GS head。GT 条件仅用于 teacher 标签生成/监督，不把 GT 先验打开给 NoPrior scene-token encoder。
3. teacher 使用 `depth-anything/DA3-GIANT`，revision `7cd62ae9315b9dff094d2d300e4ad012640607dd`。权重 SHA256 `1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001`。已有模型文件直接复用，本轮只补齐 config.json。
4. 注意官方 `extract_depth.py` 当前默认是 `DA3NESTED-GIANT-LARGE`，与论文的 DA3-Giant 描述不完全相同。本轮显式选 DA3-Giant 并固定版本，不能把两者混称。采用该脚本的 round-robin 分组思路，**每组最多 16 帧**，每组覆盖较长轨迹；具体帧组全部写入 provenance。该上限是本项目显存设置，论文未给出，组大小改变可能影响伪深度。
5. 将 teacher 输出深度双线性恢复到标准化原图尺寸，再用官方 `ImagePreprocessor` 同步处理 RGB / depth / K 到 252×252。RGB 使用其默认双线性抗锯齿，depth 使用最近邻；252 能被 patch=14 整除。
6. RGB 用无损 WebP；depth 用官方 log-uint8 WebP 和逐帧 d_min/d_max。这次先缩至 252 再量化，避免存整套高分辨率 RGB/depth；它与先量化全分辨率再读取缩放不是逐值完全相同的路径。分位数范围外约 0.2% 的像素记为无效，不能把有效率当成深度精度。
7. 源数据带畸变参数。本版沿用官方 converter 的 pinhole 处理，不额外去畸变，并明确记为 `undistorted=false`。将来改变畸变处理时需给 RGB/K/depth 整体升版本，不能混入现有标签。

manifest.json 固定版本、划分、teacher、预算；splits.json 保存完整分区；provenance/*.json 保存每帧源路径/SHA256、相机源 JSON 哈希、teacher 分组和 tar 哈希。脚本仅允许匹配 manifest 的续跑，遇到未知已存在 shard 会拒绝覆盖。

## 4. 读取和验证入口

- `tools/scene_token/prepare_training_pilot.py`：确定性选场景、生成 depth、写官方 tar。支持 `--limit 1` 先处理一个，再用相同参数恢复到完整 10 场景。
- `tools/scene_token/validate_training_pilot.py`：逐帧完整解码、划分交叉检查、2/8/16/24 context 的全场景 DataLoader、2 worker、固定验证重复性；`--backward` 加载真实 ZipSplat 权重，执行 S256、2 context + 4 target、252px 渲染与完整损失反传，无 optimizer update。
- 数据配置 `splatfactory/configs/data/dl3dv_scene_token_pilot.yaml`。其中 `test_shard_dir` 明确指向自定义 validation 目录；这是当前 loader 对 split=val 使用的字段名，**不是最终 test**。`val_overrides` 仅在构造 validation dataset 时应用，训练仍使用原 bounded sampler。
- 修复 `TensorWrapper.__torch_function__` 的 torch.stack 递归：原代码转调 tensordict.stack 又回到 torch.stack，导致真实 Camera/Pose collate 失败且场景被静默跳过。改为堆叠 packed tensor 后重建原类型，处理 tuple、负维度和 out；新增回归测试。

运行位置均为 vllm1 的 `/root/zipsplat/ZipSplat`，Python 使用该目录 `.venv/bin/python`。只同步本次明确改动的文件，不做整库同步。

## 5. 实际验证状态

本轮已完成全部处理和验收，服务器 `summary.json` 中 `complete / validation_passed / full_model_backward_passed` 均为 true。详细结果见同目录 `validation.json`；本地副本在 `ZipSplat/results/scene_token/pilot_20260907/`（Git 忽略）。

| 项目 | 实测结果 |
|---|---|
| 训练 / 验证 | 8 / 2 场景，2,640 / 686 帧，共 3,326 帧 |
| tar 总量 | 268,912,640 bytes，约 269 MB / 256.5 MiB |
| RGB / 深度编码总量 | 226,437,376 / 36,927,366 bytes，另有 tar/元数据开销 |
| 逐场景转换总耗时 | 332.86 秒，约 5.55 分钟；不含初次加载模型/检查/中途开发停顿 |
| 逐帧验收 | 3,326 对 RGB / depth 全部解码；相机/位姿/深度有限，尺寸 252×252；每帧有效深度约 99.8% |
| 训练 DataLoader | 2/8/16/24 context 各读取全部 8 场景，每个 batch 4 target；未跳过场景 |
| 多 worker | 2 worker 读取 8 个训练场景，无重复/缺漏 |
| 固定验证 | 2 个验证场景，6 context + 8 target 互斥；重复读取结果一致 |
| 真实模型 | released checkpoint + S256，2 context / 4 target，全 252px 渲染，16,384 个 GS |
| 完整损失反传 | RGB / LPIPS / Chamfer / depth 和总损失有限；scene tokens、颜色投影、融合层和高斯 head 梯度非零且有限 |
| 单次前后向诊断 | 约 1.294 秒，peak allocated 7.39 GiB；热身冻结策略、B=1、无优化器，不是完整训练吞吐/显存估计 |
| 回归检查 | 8 项 unittest 通过，包括新增 Camera/Pose 堆叠和真实 collate 问题回归 |
| optimizer update | 0；未开始拟合，不提供质量提升结论 |

已查看 RGB / 伪深度示意图 `rgb_depth_preview.png`，主体边界和远近分布基本对应；这是定性检查，不是标签准确度评测。有效率主要受编码分位数裁剪影响。

另记录一个源轨迹现象：验证场景 `59bf8258...` 全轨迹最大/中位相邻位移比约 70.37，出现在第 276→277 个零基注册帧处；固定验证只使用 0–149 帧，窗口内比值约 5.44，低于原 loader 的 10 倍过滤阈值。这轮没有禁用该过滤器，也没有把全轨迹称为已验证无异常。后续扩大取帧窗口或转换全量候选时需继续记录位姿跳变及实际跳过率。

## 6. 后续顺序

先以这批数据做少量 optimizer step / 小样本过拟合，检查 loss 下降、独立验证、保存恢复与参数分组；随后验证完整模型 DDP 和真实吞吐，再扩大到数百场景，最后按存储和训练预算扩充。8 个训练场景只用于排错，不足以评价 scene token 泛化或与原模型公平比性能。尚未验证的全量候选在转换时继续逐图检查，异常记录后跳过，不删除原数据。

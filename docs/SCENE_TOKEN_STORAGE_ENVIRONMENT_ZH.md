# 存储预算、数据跳过策略与训练环境配置

日期：2026-09-07。用户本轮询问存储/异常数据，并授权配置、更新、重装环境。本轮只调整ZipSplat环境，补充缺失的TensorBoard依赖声明；未修改模型结构、训练超参、QuerySplat环境或原始数据，未批量转换数据/生成深度，也未执行优化器更新。

## 存储预算

服务器`/data`当前剩余约3.2T；根文件系统另有约3.1T，但预算不依赖把两者容量合并。采用252×252派生训练数据，避免再次复制完整原分辨率图像。较小派生分辨率适用于本轮固定252、方形输入的实验；以后变更分辨率/裁剪增强需重新评估，原图保留。

实际在内存中使用训练转换所用的中心/主点裁剪、resize和无损WebP编码：随机seed256抽取88个DL3DV场景的首/中/末帧，共264张；RE10K train/test各12个torch文件，共315场景、43162帧统计，每文件前3场景抽首/中/末帧，共216张RGB编码。没有保存整套新图像。

| 范围 | 帧数口径 | RGB估算 | 深度预算 | tar/元数据预算 | 合计 |
|---|---|---:|---:|---:|---:|
| DL3DV 10337个候选 | 精确注册帧3530111 | 248GB | 58–231GB | 7GB | 约0.31–0.49TB |
| RE10K train+test | 由315场景抽样外推约1005万帧，非全量精确计数 | 527GB | 165–658GB | 21GB | 约0.71–1.21TB |
| 两者 | 上述范围之和 | 775GB | 223–890GB | 28GB | 约1.0–1.7TB |

GB/TB为十进制。RGB均值分别70177/52483字节每帧。深度尚未生成，16–64KiB/帧是保守排期情景，不能称为实测深度压缩率；原格式是log-u8+WebP与每帧范围，并非原尺寸float32深度。tar预算按约2KiB/帧估算。正式转换后以文件大小替换估算。

按上界留20%浮动并另留100–200GB用于checkpoint/临时分片，完整两数据集大致可按1.3–2.3TB规划。官方DL3DV测试135场景的派生数据还会有少量额外开销，包含在余量内。仅先做DL3DV则明显更省空间。当前磁盘有条件容纳，前提是直接生成所需252分辨率、控制checkpoint保留数量，不先建立全量960P中间副本。深度生成可读取较高分辨率原图，最终保存与训练视野匹配的深度，不能为了省存储悄然改变teacher推理口径。

## 数据跳过策略

- 原下载失败清单96个保持跳过。
- 旧完整清单10485个中，4个缺图、4个注册帧不足28、5个首个注册帧短边小于252，共13个独立场景，约0.124%；这13个均在训练候选内，暂跳过。
- 包括原96个在内，共109/10581≈1.03%已知需排除/暂不使用。原文件不删；4个短序列仍可能用于较少视角训练，暂时排除是为了简化首轮配置。
- 排除上述项和135个可用官方测试场景后，留下10337条候选。官方测试保留不是数据质量失败。派生清单为`training_candidates_252.txt`；它不是最终train/val划分，也不是全量逐帧解码通过证明。
- 480×270、270×480等短边足够的图像不因低于“960P”就全部丢弃；转换时还需检查实际裁剪后尺寸、所有采样帧的可解码性和相机参数。新增失败应跳过并留日志，不能静默修改统计。
- 畸变系数非零本身不是坏数据，不应因此丢弃全部DL3DV；应统一图像/相机/teacher的去畸变或针孔近似策略。

## 已完成的环境调整

环境路径：`/root/zipsplat/ZipSplat/.venv`，Python3.12.4。配置前后完整包列表和约束都保存到本轮报告。先做依赖解析dry-run，再安装；本轮已有包仅3项改变，其余新增108项（包括DA3声明的工具依赖）。

| 项目 | 结果 |
|---|---|
| NumPy | 2.5.1→1.26.4，满足ZipSplat/DA3的`numpy<2` |
| OpenCV headless | 5.0.0.93→4.11.0.86；安装同版本opencv-python满足DA3声明 |
| gsplat | PyPI1.5.3→官方推荐fork（包版本1.4.0） |
| torch/torchvision/xformers/CUDA依赖 | 保持既有版本，torch2.13.0+cu130、xformers0.0.35 |
| 训练/深度工具 | 补齐h5py、pandas、albumentations、wandb、TensorBoard、Depth Anything 3及其依赖 |

gsplat来源`https://github.com/JoannaCCJH/gsplat`，commit `5791713f43935a9093c61b21febe8c98b84607d9`；源码`/root/zipsplat/dependencies/gsplat-activated`，使用虚拟环境内CUDA13、H100 sm90、MAX_JOBS=8编译通过。包版本号较小不代表缺少所需功能，实际验证了activated输出。

DA3来源`https://github.com/ByteDance-Seed/Depth-Anything-3`，commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`；源码`/root/zipsplat/dependencies/Depth-Anything-3`。只装基础依赖，没有装会覆盖gsplat的`gs/all` extras。`DepthAnything3` API可导入；本轮没有下载脚本指定的完整DA3NESTED teacher权重，权重/配置与深度推理仍在首次小批数据准备时验证。

训练入口还发现上游requirements漏列TensorBoard，而`summary_writer.py`直接导入它。本轮补装并在本地/远端`splatfactory/requirements.txt`添加该项，没有改动已有模型代码。

## 验证结果

- `uv pip check`：195个包，依赖约束全部通过。
- `python -m splatfactory.train --help`：成功。
- `DepthAnything3` API导入成功。
- scene-token 6组单元测试通过，覆盖FP16/BF16梯度等。
- 新fork的32px合成三高斯场景：前方可见两点activated=1，背后一点=0；RGB/LPIPS/depth/Chamfer各损失有限且总损失反传通过。
- 单独检查Chamfer位置梯度：已激活两点严格为0，未激活点非零，证明官方head的detach策略实际生效。
- 真实checkpoint、S256、2视角、BF16、64pxRGB渲染损失反传通过，各必查模块梯度有限且非零。此为环境回归，不是训练质量/收敛/正式吞吐结果。
- 无优化器更新，无完整数据训练。检查结束GPU均空闲。

## 给后续开发的解释

渲染器把模型输出的三维高斯和相机转换为二维图像；gsplat是执行这一步及其反向梯度计算的CUDA库。模型预测GS→gsplat生成图像→与GT比较→反传到网络。渲染器本身不是需要训练的新backbone。当前fork额外标记参与成像的高斯，使ZipSplat将特定几何约束集中用于缺少图像监督的高斯；原PyPI版能够显示结果，但不能提供这项训练标记。

原训练采样器混合输入视角重建监督和新视角监督，不等于原实现必然错误。2026-09-07后续用户要求监督采样沿用原文/官方实现，现已确定训练保留原bounded sampler并允许context-target重叠；严格互斥用于固定持出验证/测试，取代此前“训练也严格持出”的建议。最终测试不参与调参。新增scene token从头学习，而已有DA3权重需较小幅度适配，应独立分组设学习率；这不要求改变S=256。当前数据划分、teacher和pilot状态见[数据pilot](SCENE_TOKEN_DATA_PILOT_ZH.md)，正式优化器训练仍未启动。

报告根`/root/multiview_compare/reports/scene_token/environment_20260907/`，本地副本`results/scene_token/environment_20260907/`（忽略）。包括storage.json、data_policy.json、training_candidates_252.txt、before/after/changes.json、requirements-lock.txt、安装日志、loss_validation.json、giant_render_backward.json。正式数据转换、teacher深度、训练划分与优化器完整步骤仍待后续准备。

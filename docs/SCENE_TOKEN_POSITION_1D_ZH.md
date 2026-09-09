# 一维槽位位置编码验证

日期：2026-09-08；`dev/scene_token`。用户要求先加入不同的位置编码验证，并明确质疑将scene token绑定二维网格的必要性。本实验保持每视角S256的一维槽位序列，不绑定图像二维锚点，不添加视角时间编号，不改变patch+scene的跨视角注意力范围。

## 1. 一维表示不要求二维槽位网格

“一维”指S个槽位按序列组织，每个槽位仍有1536维特征。二维网格是可选的空间先验，不是表示场景的必要条件；原图patch保留其二维位置，潜在槽位不必与某个图像格子或三维位置一一对应。

核对服务器GVC1D源码，commit `efdff7992c7cd6fbe76b05d3a5762b77d57e929b`：

- `modeling/GVC1D.py:81`创建`[num_latent_tokens,width]`可学习latent。
- `modeling/modules/blocks.py:116`另建同形状`latent_token_positional_embedding`；line234在输入时相加，随后对包含图像/latent/条件token的序列执行`ln_pre`。
- 潜在槽位不逐个绑定二维patch格子。不过其图像处理仍有二维window划分与patch位置，global模块也保留window空间组织；不能把它说成整网完全没有空间结构。

本轮不是GVC1D的完整移植：采用**持续作用于注意力Q/K的一维RoPE**，目的是独立验证槽位身份编码的作用。相比只给一个自由可学习向量再加一个同形状可学习向量，它能在深层继续提供身份信息；与GVC1D的加法embedding/ln_pre/视频条件路径不同。

## 2. 预先确定的三组实验

| 组 | Scene的1D位置 | 含义 |
|---|---|---|
| none | 不进行旋转 | 当前模型对照 |
| shared | 全部为127.5 | 改变Q/K但不区分槽位的控制组 |
| distinct | 槽位索引0至255 | 不同槽位不同编码，跨视角共享同一索引规则 |

每个DA3 block的QK norm之后、已有二维RoPE之前，对scene Q/K的head向量做旋转。base10000、scale1，所有40层生效，包括原来尚未启用二维RoPE的前13层。只旋转scene，camera/patch QK不动，V与残差流不直接加入位置向量。

旋转在实数算术下保持向量范数，BF16有舍入误差；因此不会像反复将大幅度位置向量叠加到残差流那样，直接人为增大特征差异。注意力读取内容会随Q/K变化，这是本实验预期干预。

1D RoPE仍具有槽位索引的相对结构，不是完全无结构的集合，也不是GT空间位置或真实时间。跨视角同编号槽位不强制对应同一三维点，所有token之间仍按原完整attention交互。

三组均从同一released ZipSplat和完全相同的新参数初始化出发，初始可训练参数SHA256逐组一致。每组**全部200步拟合同一个固定书架batch**，而不是旧v3的100步单batch+100步八batch。因此本轮各组之间可比较，不将旧v3的200步指标混进对照。

- 训练：书架7e6564c7…，context186/194、target187/189/190/193，零基；252px、V2/B1/4target，沿原完整RGB/LPIPS/几何/深度损失。
- 已有DA3冻结，slots+颜色query和已有下游训练；新参数LR3e-4、原下游3e-5，AdamW原参数、10步warmup、clip1、BF16；位置编码无新可训练参数。
- 验证：原固定2场景，各V6+8个互斥target，NoPrior、GT相对相机/context深度尺度，无配准/TTO，target特征不进入重建。
- 每组step0/100/200评估并存渲染；step0/200对train0与val0跑全40层诊断。只报告该种位置设计、一个训练样本、单种子和200步预算的结果。

## 3. 实现与验收

实现作为独立实验adapter：[position_1d.py](../tools/scene_token/position_1d.py)，训练工具[validate_position_1d.py](../tools/scene_token/validate_position_1d.py)。未修改正式模型源码或公共推理API，不自动启用为生产默认值。

实验checkpoint新增持久化`backbone.backbone.scene_position_experiment_spec`标记，并在`conf.experiment`保存mode/base/scale/版本与安装要求。恢复时必须先以对应配置安装adapter再严格加载；未安装的普通模型严格加载会拒绝额外标记。不能通过`strict=False`或忽略键的flexible loader将其当作无位置编码checkpoint推理。

解析检查已通过：非scene token逐值不变、旋转范数保持、相同输入在不同槽位产生不同Q/K、跨视角同槽位编码一致、梯度有限、同槽位Q/K内积保持、SDPA直接反传与非重入checkpoint重计算的输出和输入梯度逐位一致。工具[test_position_1d.py](../tools/scene_token/test_position_1d.py)。

逐层诊断仍做有/无观测hook的完整输出对照，位置adapter在两次前向中均开启。训练中检查关键梯度有限非零、优化器step连续与已有冻结参数哈希。checkpoint检查工具[check_position_checkpoint.py](../tools/scene_token/check_position_checkpoint.py)使用新建模型安装adapter后严格加载，复核固定输入推理；不等同于又做了一次优化器恢复训练测试。

## 4. 产物

- 权威目录：`vllm1:/root/multiview_compare/experiments/scene_token/position_1d_20260908_v1/`
- 本地结果：`F:/Development/multiview/ZipSplat/results/scene_token/position_1d_20260908_v1/`（Git忽略，checkpoint不下载）
- 日志：`/root/multiview_compare/logs/position_1d_20260908_v1.log`
- 每组目录含config/evaluations/steps、各检查点渲染、train/val逐层JSON与最终checkpoint；根目录保存protocol/provenance/summary/completion/reload_check、quality/layer CSV及3张对照图。

## 5. 完成结果：深层分化改善，重建质量仍未通过

三组各完成200次更新，共600次；梯度有限非零，优化器步数连续，已有冻结参数哈希不变。12组完整逐层观测（3组×初始/200步×train0/val0）的有/无诊断hook输出max_abs均0；none adapter启用前后亦逐位相同。

| 组 | 训练PSNR ↑ | 验证PSNR ↑ | 验证LPIPS ↓ | 训练总loss | 验证总loss |
|---|---:|---:|---:|---:|---:|
| none | 13.284 | 8.147 | 0.7441 | 0.21926 | 0.41578 |
| shared | 13.786 | 8.638 | 0.7363 | 0.20492 | 0.39479 |
| distinct | 13.804 | 10.899 | 0.8262 | 0.20352 | 0.30760 |

distinct相对none验证PSNR增加2.753dB，相对shared增加2.261dB；但**验证LPIPS反而更差**，相对none增加0.0820。两个验证场景PSNR分别是11.106/10.693，none为7.415/8.878；改善并非只来自某一个场景，但仍只有两个样本。

人工检查渲染：none主要为近乎常量的大色块；shared和distinct出现更多变化，但仍主要是大色块、细长尖刺/线状高斯伪影及少量噪点。书架书脊/架子细节、验证场景的可靠几何没有恢复。PSNR提升不能被表述为感知质量或高精度重建成功。

200步、书架样本、每视角256槽位统计：

| 位置 | none：cos / 有效秩 / 相对差异 | shared | distinct |
|---|---|---|---|
| block0输出 | 0.9873 / 120.79 / 12.00% | 0.9879 / 97.23 / 12.01% | 0.9838 / 64.03 / 14.00% |
| block39输出 | 0.9885 / 4.60 / 10.69% | 0.9858 / 3.69 / 11.89% | **0.8532 / 21.31 / 38.86%** |
| 几何融合后 | 0.9903 / 3.34 / 9.90% | 0.9775 / 3.31 / 14.75% | **0.9561 / 7.05 / 20.59%** |
| 颜色输出 | 0.9998 / 3.08 / 1.54% | 0.9992 / 2.24 / 2.82% | 0.9959 / 6.16 / 6.45% |

第一层的共同分量占优仍存在；持续1D编码主要改善了中后层分化，并没有普遍提高所有层的有效秩。未训练的distinct在末层已有有效秩19.19（none3.85），说明一部分多样性直接来自身份编码引导的不同读取方式，不能全部当成训练获得的场景信息。

从末层到几何融合/颜色输出，差异仍有明显收缩。可以继续在该路径定位与适配，但本轮没有额外执行下游结构替换或DA3解冻训练。

## 6. 工程检查、开销与下一步

新建模型恢复三组checkpoint均通过：未安装adapter时严格加载拒绝，正确安装后严格加载成功，固定样本loss/PSNR在预设1e-3/1e-2容差内复现；不是逐位确定性或额外优化器恢复测试。首次恢复检查工具把历史清单中的浮点帧号直接写入eval索引，导致DataLoader跳过样本；已只在检查工具中转换成整数，第二次检查通过。三组训练均使用先前验证过的真实缓存batch，这个检查工具错误不影响训练或主结果。两份恢复检查日志保留。

H100单步中位时间none0.207s、shared0.252s、distinct0.250s。实验hook实现约有21%额外开销，峰值显存约9.01/14.49/14.48GiB（包括验证/逐层诊断，非纯训练峰值）；未做正式性能优化，不据此声称1D位置编码的必要成本。三个checkpoint共约22.1GB，保存在服务器，不下载本地。

决策：**保留一维槽位设计和实验开关，当前不引入二维锚定；位置差异对深层表征有正面信号，但尚未通过重建验收，不能将其设为已验证的最终方案。** 后续先比较受限DA3适配及下游融合如何保留图像相关差异，以固定样本细节与held-out LPIPS/PSNR共同验收；如果比较GVC1D式加法可学习位置embedding/ln_pre，则独立列为另一种编码方案，不混称本轮已复现GVC1D。

仍待验证：不同位置尺度/频率或可学习身份编码、更多随机种子、更多场景/训练步数、联合DA3微调、跨视角scene-only注意力。没有自动启动这些实验，没有提交/推送。

![重建对照](../results/scene_token/position_1d_20260908_v1/reconstruction_comparison.png)

![训练与验证指标](../results/scene_token/position_1d_20260908_v1/training_curves.png)

![逐层表征对照](../results/scene_token/position_1d_20260908_v1/layer_comparison.png)

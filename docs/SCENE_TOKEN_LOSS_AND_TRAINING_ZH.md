# Scene token 损失函数与训练方法说明

版本日期：2026 年 9 月 8 日

本文用于解释当前 DA3 scene token 三维重建实验的训练目标、计算方式与梯度作用，并对照 ZipSplat 和 GVC1D 的公开训练方法。当前损失由图像重建、感知相似度、几何监督及高斯参数正则组成；没有直接约束 scene token 多样性、正交性或码率的损失。

当前实现口径以 `multiscene_20260908_v2/train_v1` 的配置、归档日志和实际源码为准。文中明确区分论文设计、公开代码及我们的小规模适配实验，避免把同名参数或日志字段当作相同训练口径。公式按单个场景书写，最后再对 batch 求平均。[1–6]

## 1 输入输出与监督信号

每个场景输入 V 张 context 图像，DA3 为每张图像加入 S=256 个可学习 scene token。当前 V=2，因此输出场景槽位总数 K=512；每个输出 token 解码 32 个高斯，合计 G=16,384 个高斯。验证 V=6 时为 1,536 个 token 和 49,152 个高斯，预算也随之增长。当前保留率为 1，scene token 输出后没有进一步减少 token 数量。

每个高斯包含中心位置 μ、三个轴向尺度 s、旋转四元数 q、不透明度 α，以及球谐颜色系数。可微渲染器将高斯投影到目标相机，产生预测 RGB 和深度；再与目标图像和伪深度比较，把误差传回高斯解码器、颜色分支及 scene token。旧 DA3 参数目前冻结，但新增 scene token 的梯度仍通过 DA3 的计算传播。

当前每场景有 4 张 target 监督图。`loss_on_context=false` 表示只对 target 集合计算图像和深度重建损失，并非训练 target 必然与 context 无交集；动态采样可能选中相同帧。预留验证帧则被单独排除于全部训练 context、target 和训练标签的 teacher 输入之外。

RGB 是数据集图像，深度是 DA3 teacher 生成的伪标签，不是传感器实测深度。相机内外参与深度共同确定监督坐标系。当前模型不把 target RGB 当作编码输入；相机和深度用于训练监督，不等于它们全部作为 backbone 的输入条件。

**坐标尺度。** 先将相机变换到所选 reference context 的坐标系，再把 context 深度反投影成三维点，取这些点到原点距离的中位数作为场景尺度。相机平移和深度除以该尺度。因此下文 0.1、1.0、10 等几何阈值均是归一化坐标中的数值，不能直接理解成米。不同 context/reference 会改变归一化结果。[3]

| 符号 | 含义 |
| --- | --- |
| T、H、W | 目标视角数、图像高与宽；当前 4、252、252 |
| I、Î | 真实与渲染 RGB，形状 T×3×H×W |
| D、D̂ | 目标伪深度与渲染深度，形状 T×H×W |
| G、M | 高斯数量、有效目标三维点数量 |
| μᵢ、sᵢ、αᵢ | 第 i 个高斯的中心、三个轴向尺度及不透明度 |
| pⱼ、aᵢ | 第 j 个目标三维点、第 i 个高斯是否被渲染器标为 activated |
| [x]₊、sg(x) | max(x,0)、数值不变但切断该路径梯度 |

## 2 总损失与有效权重

$$
L=L_{\mathrm{RGB}}+0.05L_{\mathrm{LPIPS}}+0.1L_{\mathrm{loc}}+0.01L_{\mathrm{depth}}+P_s+P_\alpha+P_\mu
$$

代码把这些项分成 `rendering_loss` 和 `geometry_loss`：前者包含 RGB、LPIPS 及三个参数惩罚，后者包含加权位置损失与加权深度损失。总损失是这两个分组相加，而不是把日志中的所有 loss 字段再加一次。[1–2]

| 日志字段 | 当前计算内容 | 进入 total 的系数 |
| --- | --- | --- |
| mse_loss | RGB 的平均绝对误差 L1 | 1.0 |
| lpips_loss | VGG LPIPS 感知距离 | 0.05 |
| location_loss | 单向 Chamfer 加内部越界惩罚 | 0.1 |
| depth_loss | 有效目标像素上的深度 L1 | 0.01 |
| scale_penalty | 高斯尺度越界平方惩罚 | 1.0 |
| opacity_penalty | 不透明度低于 0.01 的平方惩罚 | 1.0 |
| location_penalty | 中心坐标超出 ±10 的平方惩罚 | 1.0 |

**配置审计发现。** 我们当前实验没有加载官方完整模型 YAML，而是从模型类默认配置加实验覆盖项构造模型。其有效 `max_scale=1.0`；官方 `splatfactory/configs/model/zipsplat.yaml` 写的是 `max_scale=0.1`。加载 released 权重只加载参数，不会自动把这项训练配置带过来。此前书架长拟合和当前多场景实验均采用 1.0，因此不能单凭这一共同设置解释两轮差距；但也不能说实验损失配置与官方完全一致。本次仅核实和记录，不追溯修改实验或训练配置。[1,4–6]

## 3 RGB 和感知损失

### 3.1 RGB 平均绝对误差

$$
L_{\mathrm{RGB}}=\frac{1}{3THW}\sum_{t,c,h,w}|\hat I_{tchw}-I_{tchw}|
$$

代码由 `use_l1_loss=true` 选择 `F.l1_loss`，对目标视角、颜色通道和像素统一平均。日志虽然叫 `mse_loss`，当前并没有计算 RGB 均方误差；只有把该开关设为 false 才会改成平方差。说明实验时应称为“RGB L1”，避免误导。

该分支对预测 RGB 不做 clip；因此超出 [0,1] 的预测值仍能得到纠正梯度。L1 强调像素颜色一致性，但无法单独保证正确几何或高频纹理。以一个像素通道为例，预测 0.7、真实 0.5，则 L1 为 0.2，MSE 为 0.04，两者不是可互换的日志名称。

### 3.2 VGG LPIPS 感知距离

$$
L_{\mathrm{LPIPS}}=\frac{1}{T}\sum_t\operatorname{LPIPS}_{\mathrm{VGG}}(\operatorname{clip}(\hat I_t,0,1),\operatorname{clip}(I_t,0,1))
$$

LPIPS 用预训练网络比较多层视觉特征，帮助约束结构、纹理和视觉相似性。当前采用 VGG 版本，网络权重固定，梯度仍会从其特征计算传回预测图像；它不是判别器，也没有对抗训练。调用 `normalize=true`，将 [0,1] 图像映射到 LPIPS 所需的 [-1,1]。

其内部可概括为：对每层空间位置的特征做通道归一化，计算对应特征的平方差，再用已学习的通道权重聚合、空间平均并跨层求和。复现数值应调用当前 LPIPS 实现及配套权重，而非用普通 VGG 特征 MSE 代替。[2]

这里先裁剪 RGB，所以越界部分不能经 clip 获得该分支的有效梯度；RGB L1 没有这个裁剪。0.05 只是该项的乘数，不能解释成它仅产生总梯度的 5%。

## 4 深度和三维位置损失

### 4.1 有效像素上的深度 L1

$$
m_{thw}=\mathbf{1}[D_{thw}\geq0],\qquad L_{\mathrm{depth}}=\frac{\sum_{t,h,w}m_{thw}|\hat D_{thw}-D_{thw}|}{\max(1,\sum_{t,h,w}m_{thw})}
$$

无效深度采用负值标记。当前在所有目标视角的有效像素上统一求平均；若不同图有效像素数不同，它们的贡献与有效像素数有关，不是各图先平均再等权平均。某个场景完全没有有效深度时，深度及三维位置监督记为零。输入标签还必须是有限数值，掩码不能自动消除 NaN 乘零造成的污染。

**渲染深度的含义。** 当前调用 gsplat 的 `RGB+D`，其中 D 是累积 z-depth，而非除以累计透明度的期望深度。按从近到远的顺序，单个高斯在某个像素上的有效不透明度记作下式的 αᵢᵖⁱˣᵉˡ：[3,7]

$$
w_i=\alpha_i^{\mathrm{pixel}}\prod_{j<i}(1-\alpha_j^{\mathrm{pixel}}),\qquad \hat D=\sum_i w_i z_i
$$

该有效值由高斯不透明度与其像素覆盖共同决定，和 activated 的二值标记 aᵢ 不同。gsplat 的 `ED` 才进一步除以 Σwᵢ。覆盖不足时，即便高斯中心深度正确，D 仍可能偏小；因此当前深度损失同时影响位置和覆盖/不透明度。不要把 D 与 ED 混用，也不要在未做对照时直接更换渲染模式。

### 4.2 从目标伪深度得到三维点云

每个有效目标像素根据相机内参反投影，再通过目标相机位姿变换到共同场景坐标系。对于标准针孔相机，可写为：

$$
p=R\left(D(u,v)K_{\mathrm{cam}}^{-1}[u,v,1]^\top\right)+t
$$

实际执行的是项目 `Camera.unproject_depth` 和 `Pose.transform`；使用时应遵从它们的相机及像素坐标约定。把所有 target 的有效点合并得到目标点云。batch 中点数不同时，用已有有效点的随机重复补齐长度；重复点不改变单向最近邻最小距离的数学值。[1,3]

### 4.3 单向 Chamfer 和梯度屏蔽

先在几何位置分支构造中心 μ̄：activated 高斯使用 sg(μ)，其他高斯使用 μ。`sg` 的含义是前向数值保持不变，反向传播时停止该分支梯度。activated 掩码取当前 context 和 target 渲染器标记的并集，并不等同于肉眼可见高斯或达到某个透明度阈值的高斯。[1,3]

$$
\bar\mu_i=\begin{cases}\operatorname{sg}(\mu_i),&a_i=1\\\mu_i,&a_i=0\end{cases},\quad C=\frac1G\sum_{i=1}^G\min_j\|\operatorname{clip}(\bar\mu_i,-10,10)-\operatorname{clip}(p_j,-10,10)\|_2^2
$$

Chamfer 底层使用最近邻的**平方欧氏距离**，对所有 G 个高斯平均。方向只有“高斯中心到目标表面”，没有反向“每个目标点都要被高斯覆盖”。这能把游离于场景表面的高斯拉回来，但不会强制每块表面分到相同数量的高斯，也不能单独保证完整覆盖。[2,7]

计算 loss 数值时 activated 高斯仍然在求和与分母中；只有其位置分支的梯度被屏蔽。因此“只对未渲染高斯算 Chamfer”是一种不够精确的说法。高斯仍可通过 RGB、LPIPS、深度及外层参数惩罚获得梯度，也不是整颗高斯被永久冻结。

### 4.4 Chamfer 内部的越界惩罚

裁剪会让大幅越界的坐标失去最近邻距离梯度，包装器因此额外加入一个基于未裁剪中心的项：

$$
E=\frac1{3G}\sum_{i,k}\sqrt{[|\bar\mu_{ik}|-10+10^{-6}]_++10^{-8}},\qquad L_{\mathrm{loc}}=C+0.1E
$$

这个 E 继承上节的 detach 规则，进入总损失后的有效系数为 0.1×0.1=0.01。注意内部参数虽然叫 `scale_penalty_weight`，惩罚的是位置越界，和高斯尺度 s 无关。平方根使较大越界量处的梯度衰减，但不能据此声称阈值附近梯度一定小；稳定项和裁剪边界仍影响梯度。

即便所有坐标都远离边界，E 也有 0.0001 的常数底值，故这一部分对 total 有约 0.000001 的常数贡献；常数部分梯度为零。它不影响最优解，但精确复算日志时应保留。[2]

## 5 高斯参数正则

三个正则都直接进入 rendering_loss，系数为 1。它们的数值都包含平均化；不能拿单个异常高斯的惩罚当作整个场景的 penalty。[1]

### 5.1 尺度越界惩罚

$$
P_s=\frac1{3G}\sum_{i,k}\left([s_{ik}-s_{\max}]_+^2+[s_{\min}-s_{ik}]_+^2\right)
$$

当前 s_max=1.0，s_min=10⁻⁶。尺度是三个轴的尺度参数，不是高斯协方差矩阵本身。解码器先通过 softplus 等参数化产生正尺度，并硬裁剪到 [10⁻⁶,15]；所以当前下界惩罚通常不触发，上界 1.0 则是软惩罚，不是硬上限。

例如某一轴尺度为 0.2：当前阈值 1.0 对它的上界惩罚是 0；官方阈值 0.1 的该轴未平均惩罚是 0.01。阈值差异可能影响大高斯和模糊倾向，但效果必须通过同初始化、同采样的单因素实验判断。

### 5.2 过低不透明度惩罚

$$
P_\alpha=\frac1G\sum_i[0.01-\alpha_i]_+^2
$$

α 越小，表示越透明。此项抑制的是“太透明”，会推动低于 0.01 的 α 增大，而不是鼓励 α 归零或鼓励稀疏。α=0 时单个惩罚为 0.0001；α=0.02 时为 0。对触发区间的导数是 2(α−0.01)，梯度下降会增大 α。不要把它写成 opacity L1 或高斯剪枝正则。

### 5.3 外层位置越界惩罚

$$
P_\mu=\frac1{3G}\sum_{i,k}[|\mu_{ik}|-10]_+^2
$$

这是独立于 Chamfer 内部 E 的另一项。它对原始预测中心计算，不使用 activated detach，因此已经参与渲染但位置越界的高斯也会被它约束。`location_penalty` 和 `location_loss` 含义不同，不能混为一项或重复计入 total。

## 6 日志复算与优化步骤

### 6.1 实际训练日志算例

以下来自当前多场景训练 rank 0、第 5000 次更新所记录的场景前向，属于该次更新前的训练 loss，不是最终模型重新评估结果，也不是八卡平均值。[5]

| 项目 | 原始数值 | 加权后贡献 |
| --- | --- | --- |
| RGB L1 | 0.1398281753 | 0.1398281753 |
| VGG LPIPS | 0.6562500000 | 0.0327148438 |
| location_loss | 0.0122763738 | 0.0012276374 |
| depth_loss | 0.2005798072 | 0.0020057981 |
| scale_penalty | 0.0000602121 | 0.0000602121 |
| opacity_penalty | 0.0000444796 | 0.0000444796 |
| location_penalty | 0 | 0 |

这里 LPIPS 的加权结果需要保留混合精度语义：实数乘法 0.65625×0.05=0.0328125，而在 BF16 中这一乘法得到 0.03271484375。按该值相加，rendering_loss≈0.1726477146，geometry_loss≈0.0032334355，合计≈0.17588115；与日志 total=0.1758811474 的差约 1.2×10⁻⁹。若直接对所有日志字段作双精度加权，会相差约 9.77×10⁻⁵，不能误判为漏了一项损失。

原始 lpips_loss=0.65625 不能不乘 0.05 就与 total 比较，也不能再把 location_loss_scaled 和 geometry_loss 都额外加进去。上表的加权列采用实际 BF16 乘法口径，其余展示保留小数。

### 6.2 从标量 loss 到参数更新

Trainer 先对本卡场景 loss 求平均，再按梯度累积次数缩放，执行反向传播。当前每卡 1 场景、8 卡 DDP，梯度同步相当于平均 8 个场景的梯度。混合精度使用 BF16；GradScaler 反向缩放后，先 unscale 梯度，再进行全局梯度范数裁剪，阈值为 1，随后 AdamW 更新。[3,5]

当前 AdamW 的 β=(0.9,0.95)，weight_decay=0.05；新增 scene token 和颜色 query 学习率为 3×10⁻⁴，其余可训练下游参数为 3×10⁻⁵。学习率先经过 10 次 scheduler 迭代从 0.1 倍升至 1 倍，之后保持不变；它不是官方的长 warmup 加余弦退火。当前参数分组按学习率匹配，没有专门对 token、bias 或归一化参数豁免 weight decay。

AdamW 的 weight decay 是优化器中的解耦衰减操作，不作为一个额外 loss 字段加入 total。各损失的标量大小也不等于其对某层梯度的贡献；判断谁主导训练，应在同一批样本上分别测各加权项对 scene token、颜色 query 和解码器的梯度范数及梯度夹角。

### 6.3 指标与当前未使用的损失

PSNR 先将预测和真实图像 clip 到 [0,1]，计算各图 RGB MSE，再取 −10log₁₀(MSE)，最后对视角平均。它是评估指标，不是当前的反向训练目标；它与未裁剪的 RGB L1 不是同一个量。因此 total 降低不保证 PSNR 每次都提高。

当前没有额外的 scene token 去相关、正交、cosine 排斥、KL、熵、码率、GAN、SSIM 训练项，也没有额外监督 DA3 自身深度头。S=256 是手动设定的表示预算，并不表示模型已学习了最小码长或自适应 token 数。颜色分支仍可读取原图颜色特征，因此重建提升本身还不能单独证明所有信息都压进了 scene token。

## 7 ZipSplat 的训练方法

### 7.1 公开训练设置

ZipSplat 论文报告：DL3DV 与 RealEstate10K 混合训练，DA3-Giant 生成深度监督，图像 252×252，每场景 4 张监督图，450,000 次更新、16 张 GH200。该规模不能与当前 8 个场景的短适配实验直接比较。[8]

公开配置可进一步核对：AdamW 主学习率 3×10⁻⁴，backbone 学习率乘 0.1；β=(0.9,0.95)，weight_decay=0.05，梯度裁剪 1，BF16。前 5% 步数 warmup，随后余弦退火到零。数据设置的 batch_size=384 配合动态视角数用于采样预算，不能无条件理解为每次都固定 384 个场景；实际场景数还取决于 sampler 的视角数和分布式划分。[4]

### 7.2 与当前方案最相关的训练技巧

**视角数课程。** 公开配置把 context 数从 2 逐渐增加到 24，进程参数为训练前一半。目的在于从较容易的少视角问题过渡到更多视角的共享表示，而不是只在测试时增加视角。我们的多场景训练始终为 V=2，尚未覆盖这一训练过程。[4,8]

**压缩率课程与随机预算。** 官方使用 query_sample_ratio=[0.5,1.0]、query_ratio_schedule=0.5、query_scale_with_views=0.5；压缩逐步加强，且视角增多时允许更低保留率。训练覆盖多个预算，有助于推理时调节 k-means 保留率。当前 scene token 实验设置 [1,1] 且关闭调度，已经在编码输出处限制到每视角 256 个槽位；二者不能按相同百分比解释。[4]

**几何辅助与选择性停止梯度。** 自由放置的高斯没有逐像素射线作为位置锚点，需要伪深度和单向 Chamfer 提供早期几何引导，同时避免这一最近邻项继续拉动已参与渲染的高斯。这是当前已继承的关键设计，不能只保留 RGB loss 就宣称训练口径不变。[1–2,8]

**高斯分组耦合初始化。** 官方代码让同 token 的一组高斯输出使用耦合初始化，再用小尺度、偏低不透明度及相机前方的位置先验降低早期渲染困难。不过我们是加载 released 全模型权重后新增 scene 参数，预训练高斯头权重会覆盖构造时初始化；不能把构造器的偏置当作当前训练开始时每个高斯的实际状态。[1,6]

**预训练 backbone 联合适配。** 官方为 backbone 设置较低但非零的学习率。当前冻结旧 DA3 是可行性阶段的受控选择，并非完整原文方案。新 scene token 改变了编码器输入分布，只训练槽位和下游能否充分适配，需要与有限 DA3 解冻单独对照。

ZipSplat 的测试时 token/camera 优化属于推理阶段的附加优化，不能作为常规训练步骤计入；当前多场景可视化未做这种优化。[8]

## 8 GVC1D 的训练方法

### 8.1 任务与阶段

GVC1D 面向视频压缩：编码一维 latent，进行熵编码/解码，重建视频帧。它没有三维高斯、目标相机渲染或我们使用的 Chamfer 深度几何项。其一维表示思想可借鉴，但好的视频重建纹理不直接保证正确的新视角几何。[9–10]

论文将训练概括为三阶段：从 TA-TiTok 预训练权重适配视频重建；加入熵模型做率失真优化并增加训练帧数；最后训练全局交互与记忆模块，处理长序列和高分辨率。使用 OpenVid 与 Vimeo 数据。[9] 官方仓库进一步将这些阶段拆成多个训练策略，不能只按一个统一 YAML 理解实际训练。

| 公开代码阶段 | 主要变化 | 代表设置 |
| --- | --- | --- |
| Phase 1 重建适配 | frame condition 适配，再放开联合训练 | 2 帧，32 latent，lr 10⁻⁴，FP16，EMA |
| Codec 初训与率失真 | 先重建适配，再打开 bpp；帧数逐步增加 | 2→3→4→6，再 7→13→32；lr 5×10⁻⁵ 等分段值 |
| Global 模块 | 先只训练新 global 模块，再联合 | 512 分辨率，短序列到长序列 |
| Memory 模块 | 逐步加入长期参考，随后联合微调 | 逐渐增到 33 帧，后期 lr 降到 10⁻⁶ 量级 |
| 最后感知适配 | 特定策略启用 GAN | 不是全部阶段一直启用 |

上表来自 `pretrain.sh`、`train.sh` 与 `get_training_strategy`。不同入口的帧数可能包含首个 I 帧；例如 33 帧序列可对应 1 个 I 帧和 32 个预测帧。不能把 shell 级阶段数、论文的三大阶段及单次实际运行步数混为一谈。[10–12]

### 8.2 重建预训练损失

Phase 1 代码配置为 MSE、感知项、VAE KL 和满足启用条件后的 GAN：

$$
L_{\mathrm{pre}}=L_{\mathrm{MSE}}+1.1L_{\mathrm{perc}}+10^{-6}L_{\mathrm{KL}}+0.1f_{\mathrm{GAN}}L_{\mathrm{GAN}}
$$

这里 MSE 是真正的像素平方差。配置 logvar=0 且冻结，所以相关 exp(logvar) 缩放为 1。KL 项约束 VAE posterior 接近标准正态，代码按非 batch 维求和再对 batch 平均，不能与按 latent 元素平均的 KL 使用同一数值权重。[10]

感知项也不等于我们的 VGG LPIPS：Phase 1 的 `lpips-convnext_s-1.0-0.1` 把 LPIPS 与 ConvNeXt-S 输出的 MSE 做加权平均。其外层再乘 1.1，因而这一配置等价于 LPIPS 加 0.1 倍 ConvNeXt 项。Phase 2 使用 `lpips_mix` 组合，不能直接拿双方感知 loss 数值比较难度。[10]

公开 YAML 的 discriminator_start=200000，即 GAN 是否参与取决于实际 global_step。`pretrain.sh` 注释写 warmup 约 5k、另一阶段约 20k，但没有将 YAML 的 max_train_steps=650000 自动改成这些数。短训练若未到 200k，不能因为配置中有 GAN 权重就说实际用了 GAN。

### 8.3 率失真与后期 GAN

Phase 2 的总体结构是：

$$
L_{\mathrm{codec}}=\mathbb{E}_{b}\left[\lambda_b\,\mathcal{D}_b+R_b\right]
$$

R 是 bpp 码率估计；λ 控制同一码率下更重视重建还是更重视节省比特，论文采用 0.07 到 1.5 的对数间隔工作点。[9–10] 固定 token 数并不等于固定 bpp，因为量化值的概率分布和熵编码效率仍会变化。

代码在相应阶段把失真项写成逐帧/逐样本加权的重建、感知及可选 GAN 组合。尤其 `cal_mix_Loss` 使用 0.5×L1+0.5×MSE，所以策略名含 `mse` 不代表实际只有纯 MSE。名为 `mse` 的初适配策略不启用 bpp，随后 `rdc_mse` 打开 bpp，而 `rdc_mse_gan` 才进一步启用对抗项。[10–11]

后期 GAN 使用自适应梯度权重：比较重建及感知项与 GAN 项对指定解码层的梯度范数，加入稳定分母、裁剪后再乘基础 GAN 系数及其他策略缩放。判别器还使用 LeCam 正则。它的有效 GAN 权重不是一个从头到尾恒定的 0.1，不能只抄 YAML 的数字复现。[10–11]

### 8.4 最有参考价值的工程方法

**新增模块先适配，随后联合训练。** 仓库根据 training_module 冻结已有编码器/解码器的大部分参数，只训练新增 frame condition、global 或 memory 模块；后续策略解除相应限制。它不是始终冻结整个原始模型，只靠随机 latent 从零学会复杂任务。[10–12]

**预训练与归一化。** 一维 latent 来自预训练 tokenizer 路线，构造时 latent 与位置嵌入采用宽度相关的随机初始化；图像和 latent 拼接后进入 ln_pre，并在输出端归一化。这能作为我们尺度诊断的参考，但不能把它的数值直接当作 DA3 scene token 的最优标准差。它没有要求一维 latent 必须绑定图片二维网格；高分辨率图像仍有分块/空间结构，32 latent 也不是任意大小场景都只需 32 个 token 的证明。[10]

**时间长度课程与分组反传。** partial cascade 在小组内保留时序依赖，组间 detach 参考帧、参考特征或记忆状态，控制长序列反向图和显存。它适用于有顺序的视频预测链；无序多视角输入不能直接按时间顺序照搬。[11–12]

**优化器与数值稳定。** Phase 1 使用 EMA；Phase 2 配置改为 BF16、关闭 EMA，梯度裁剪阈值为 0.2。AdamW 配置 β=(0.9,0.999)、weight_decay=10⁻⁴，并在参数分组中排除 bias、norm、latent/embedding 等的衰减。我们当前所有可训练分组统一使用 0.05 衰减，后续可单独评估这种分组差异，但本次不据此判断其为效果瓶颈。[10–12]

## 9 对后续 scene token 训练的建议

**首先把训练目标固定并可复算。** 下轮应显式写入全部有效 loss 配置，记录 max_scale、LPIPS 类型、D/ED、context 是否监督、detach 规则和坐标归一化。将 scale 阈值 1.0 与 0.1 作为独立变量比较，避免一边更改采样和解冻范围、一边静默改阈值。

**先解决受控拟合与共享适配，再判断容量。** 保留用户选定的 std=0.02 和 S=256，固定初始 state_dict、context/target 与 reference，在 8 场景固定样本上检查收敛；再与动态采样对照。这能拆分每帧监督量与跨场景共享模型适配难度。当前 loss 标量本身不能证明 token 数不足。

**借鉴分阶段联合微调。** 在新增 scene token、颜色 query 和下游已适配的基础上，单独测试有限 DA3 解冻和较低 backbone 学习率。不要同时加入 GAN、正交正则和新注意力结构，否则很难定位收益。冻结或解冻是实验变量，不应先验保证哪一个必胜。

**在训练中逐步增加视角。** V=2 可作为起点，但检验多视角训练应真正训练 V=4/6 等配置，而不是仅拿 V=2 checkpoint 改成 V=6 推理。固定每视角 S 时总 token/GS 数随 V 增长，报告时须披露；若要分离纯视角信息增益，应另做相同总输出预算的对照。

**用梯度诊断指导 loss 调整。** 分别记录加权 RGB、LPIPS、深度、Chamfer 和参数正则对关键模块的梯度；同时记录 activated 比例、低透明度高斯比例、尺度分布和渲染累计透明度。这样才能辨别细节模糊来自监督不足、覆盖不足、过大高斯或优化冲突。

**暂不优先增加 GAN 或码率损失。** 当前目标是几何和新视角重建的可靠性。GAN 可能改善观感却产生跨视角不一致纹理；码率损失则需要真实量化/熵模型才有明确意义。可先评估温和感知权重对照，但保持 held-out 视角、几何和预算指标共同验收。以上均为后续建议，本次没有启动训练、改变模型或修改历史结果。

## 10 来源与复核入口

以下代码路径按仓库根目录书写。当前 ZipSplat 存在未提交的 scene token 实现，因此 HEAD 不代表完整工作树；对应损失、配置与日志应结合本次审计哈希和实验归档阅读。

1. 当前高斯头：`splatfactory/models/decoders/gaussian_head.py`，重点 `_rendering_loss`、`_penalties`、`_geometry_loss`、`loss`。
2. 当前损失实现：`splatfactory/models/losses.py`，重点 `ChamferLoss`、`LPIPSLoss`。
3. 当前数据与训练：`splatfactory/datasets/multi_view_dataset.py`、`geometry/scale.py`、`gaussians.py`、`trainer.py`、`models/metrics.py`。
4. ZipSplat 公开训练配置：`splatfactory/configs/zipsplat.yaml`、`configs/train/zipsplat.yaml`、`configs/model/zipsplat.yaml`；模型默认值另见 `models/networks/zipsplat.py` 与高斯头。
5. 当前实验记录：`results/scene_token/multiscene_20260908_v2/train_v1/config.json`、`steps_rank0.jsonl`、`provenance.json`；权威实验存于 vllm1 的 `/root/multiview_compare/experiments/scene_token/multiscene_20260908_v2/train_v1/`。
6. 权重加载：`splatfactory/models/base_model.py`；历史书架实验 `scale_longfit_20260908_v1`。
7. 实际环境包：vllm1 ZipSplat `.venv/lib/python3.12/site-packages/gsplat/rendering.py` 与 `chamferdist/chamfer.py`；[gsplat 渲染文档](https://docs.gsplat.studio/main/apis/rasterization.html)。
8. [ZipSplat 论文 v1](https://arxiv.org/html/2606.05102v1)，第 3.3、4.1 节及附录 A。
9. [GVC1D 论文 v1](https://arxiv.org/html/2603.15302v1)，方法部分与附录 A。
10. GVC1D 固定版本 `efdff7992c7cd6fbe76b05d3a5762b77d57e929b`：[损失实现](https://github.com/GVC1D/GVC1D/blob/efdff7992c7cd6fbe76b05d3a5762b77d57e929b/modeling/modules/losses.py)、[感知模块](https://github.com/GVC1D/GVC1D/blob/efdff7992c7cd6fbe76b05d3a5762b77d57e929b/modeling/modules/perceptual_loss.py)、[编码器模块](https://github.com/GVC1D/GVC1D/blob/efdff7992c7cd6fbe76b05d3a5762b77d57e929b/modeling/modules/blocks.py)、[Phase 1 配置](https://github.com/GVC1D/GVC1D/blob/efdff7992c7cd6fbe76b05d3a5762b77d57e929b/configs/phase1/config.yaml)、[Phase 2 配置](https://github.com/GVC1D/GVC1D/blob/efdff7992c7cd6fbe76b05d3a5762b77d57e929b/configs/phase2/config.yaml)。
11. GVC1D [训练策略代码](https://github.com/GVC1D/GVC1D/blob/efdff7992c7cd6fbe76b05d3a5762b77d57e929b/scripts/train_tatitok.py) 与 [优化器和训练循环](https://github.com/GVC1D/GVC1D/blob/efdff7992c7cd6fbe76b05d3a5762b77d57e929b/utils/train_utils.py)。
12. GVC1D [pretrain.sh](https://github.com/GVC1D/GVC1D/blob/efdff7992c7cd6fbe76b05d3a5762b77d57e929b/pretrain.sh)、[train.sh](https://github.com/GVC1D/GVC1D/blob/efdff7992c7cd6fbe76b05d3a5762b77d57e929b/train.sh)。脚本注释、配置上限与实际运行步数须分别理解。

本次审计与排版工作目录：`results/scene_token/loss_training_review_20260908/`。文档正文及公式是对上述实现的说明；其中后续实验设计属于建议，并非已完成的对照结论。

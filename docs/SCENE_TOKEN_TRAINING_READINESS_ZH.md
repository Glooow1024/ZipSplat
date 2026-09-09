# S=256 训练准备检查

日期：2026-09-07。分支：`dev/scene_token`。本轮检查数据、依赖与训练口径，没有训练模型、安装依赖或批量生成深度。原始数据和既有模型代码保持不变。

后续状态：用户随后授权调整环境；缺包/NumPy冲突和gsplat activated缺失已解决并验证，候选清单细化为10337，存储预算已抽样测量。详见[存储与环境配置](SCENE_TOKEN_STORAGE_ENVIRONMENT_ZH.md)。下文保留本次审计时的历史状态；数据转换/深度和正式训练配置仍未完成。

## 1. 结论

硬件与基础模型已具备，原始数据也足够开始研究；目前不能直接启动正式训练。阻塞项是训练 shard/深度尚未接入、训练依赖不完整、当前渲染器缺少官方几何损失需要的标记。完整训练配置与独立验证划分也尚未冻结。

## 2. 数据现状与划分

| 数据 | 检查结果 | 后续处理 |
|---|---|---|
| DL3DV-10K | `/data/datasets/3dvision/DL3DV-10K`，有效清单10485，异常96 | 使用有效清单，转换前明确剔除测试集 |
| 官方DL3DV测试划分 | 代码`split.json`有141个ID，其中135个在有效清单、6个缺失 | 135个不进入训练；不声称官方测试集完整 |
| 训练候选 | 排除上述测试ID后10350个；再排除本轮8个被标记场景，暂得10342个 | 仍需分辨率/图像质量筛选，再留独立validation；不是已冻结训练集 |
| DL3DV-Evaluation | `/data/datasets/3dvision/DL3DV-Evaluation/images`，55场景，与有效10K及官方测试ID均不重叠 | 已用于预实验的9场景仍作为开发参考，不当作全新测试 |
| RE10K | `/data/datasets/RE10K/re10k_torch`，train索引66033场景/4866个torch文件；test索引7286场景/543文件 | 两个索引之间0重叠，所引用文件均存在；仍须转为ZipSplat tar |
| RE10K `train_tt` | 索引与test的SHA256完全相同，7286场景 | 不当作训练集；目录名不能作为划分依据 |
| 另一份DL3DV | `/data/datasets/DL3DV/dl3dv_torch_960`，分桶索引共9247个唯一场景，与上述测试集合0重叠 | 可作格式转换参考/备选；首个抽样为960×540、330帧、无depth字段，不能直接当作就绪训练数据 |

RE10K train/test现有torch文件总大小约575GB（十进制），不需要重新下载原始数据。官方RE10K转换器输入是ZIP内部的torch，当前已有解包目录；应增加目录读取入口，避免为了适配脚本重新下载或重新打包整个数据集。

### 全量元数据/文件存在性复核

已读取10485个`transforms.json`，检查3580526条注册帧记录的图像存在性/非空、相机内参有限值和正焦距、位姿矩阵形状/有限性/齐次行及旋转正交误差。旋转正交最大误差约2e-15。图像文件总大小约3.014TB（包含场景目录中未注册图像）；不宜在剩余3.2T的`/data`再直接复制全量原分辨率数据。

旧完整清单中8个场景不满足本轮严格条件，全部属于训练候选：

- 4个场景缺少对应注册图像，分别2、1、90、127帧，共220帧；其中前两个在已有960P源ZIP中也缺相同帧，另外两个的源包在本次检查路径不存在，尚未判定缺件原因。
- 4个场景仅10、16、15、19个注册帧，不足24context+4target。它们可能仍适用于较低视角数，并非因此判定下载损坏；首轮先排除以固定可用视角范围。
- 未改写共享原始完整/失败manifest；逐场景记录见`flagged_scenes.tsv`。`training_candidates_provisional.txt`为10342条暂定候选，尚未冻结train/val，也未完成分辨率筛选。

另外检查了每个场景第一个注册图像的文件头：10032个为960×540，429个480×270，7个540×960，1个704×396，10个270×480，4个320×180，1个160×90，1个因首帧缺失无法读取。不能把目录名“960P”当作所有样本实际分辨率；小于252短边的图像需排除或明确上采样口径。该检查只覆盖每场景首个注册图像，不保证场景内所有帧尺寸一致。22张分桶抽样RGB完整解码成功；不是358万帧全量解码/CRC检查。

RE10K抽取train/test的首末torch文件，共4个文件，均可安全加载；首场景相机有限、RGB能解码为640×360、没有depth字段。索引存在性已全查，内容完整性仍是抽样结果。

DL3DV转换器会扫描输入目录全部ZIP，并在部分帧缺失时跳过帧；它不自动消费本项目`MANIFEST_COMPLETE.txt`。准备阶段需显式过滤有效清单并核对转换前后场景/帧数量，不能直接对包含96个异常包的目录全量运行。

当前ZipSplat默认`data/`不存在，未找到已接入其训练入口的`train-scenes/test-scenes` tar和配套深度。原始RGB/相机数据齐全不代表可直接启动训练。

## 3. 深度与相机口径

- 正式数据管线需要context/target的深度：既用于几何监督，也用于场景尺度归一化；不能用全零占位或关闭损失掩盖缺项。
- 当前ZipSplat虚拟环境没有`depth_anything_3`。服务器缓存有DA3-GIANT的模型文件，但当前深度脚本硬编码的是`depth-anything/DA3NESTED-GIANT-LARGE`；该变体在本次检查的HF缓存中不存在。生成前应明确teacher版本、模型配置和权重hash，不能默认为现有backbone权重就足够。
- 深度脚本默认`max_views=None`，即整段场景一起推理。全量准备前必须先小批次测定分块视角数、显存、时间与有效深度覆盖率，再固定分块/尺度对齐口径。
- 全部10485个场景的元数据都含至少一个非零畸变系数。转换器执行OpenGL→OpenCV和裁剪缩放后的内参变换，但未做显式镜头去畸变；生成teacher深度前应固定去畸变或针孔近似方案，并保证图像、内参、teacher与训练渲染口径一致，不能只缩放K就认为已处理畸变。
- 建议用独立派生数据目录保存shard/深度，不改共享原始目录。先处理小规模代表样本并估算磁盘，再扩大；不要先制造一份全量960P副本。

## 4. 环境与资源

| 检查 | 状态 |
|---|---|
| GPU | 8×H100 80GB，NVLink互联；检查完成后全空闲，不代表预留 |
| 多卡基础通信 | 8进程NCCL、BF16 all-reduce通过；不等于完整模型DDP验证 |
| 内存/磁盘 | 系统约2TiB RAM；`/data`余3.2T，根文件系统余3.1T；共享内存512G |
| Python | ZipSplat专用`.venv`，Python3.12.4，torch2.13.0+cu130 |
| 初始化checkpoint | `/data/venvs/.torch_cache/hub/zipsplat/zipsplat-da3g-252p.tar`存在，5791182337字节；加载兼容性沿用P2结果 |
| 核心代码一致性 | 本地/远端8个scene-token源码/配置文件归一化换行后SHA256一致 |
| 正式训练入口 | `python -m splatfactory.train --help`在导入阶段因缺`h5py`失败 |
| 缺少的包 | 实测缺`h5py`、`pandas`、`albumentations`、`wandb`；深度工具另缺`depth_anything_3` |
| 已有主要组件 | torch/torchvision、Hydra、chamferdist、fused_ssim、LPIPS、plotly、sklearn、tensordict等可导入；不等于正式总损失已验证 |
| 依赖版本冲突 | `uv pip check --python .venv/bin/python`发现现有numpy2.5.1不满足zipsplat声明的`numpy<2`；不在检查阶段直接降级，后续需完整解析和回归验证 |
| 渲染器 | 当前gsplat1.5.3能完成32×32单高斯RGB+ED渲染，输出有限，但实测info没有`activated` |

`activated`缺失是训练语义问题：官方head用它阻断已由图像监督覆盖的高斯所接收的Chamfer梯度。当前代码会警告后退化为空mask，使该策略失效。正式训练前需验证官方推荐gsplat fork及其CUDA编译/反传兼容性；不能把P2的RGB反传通过当作完整几何损失等价。

## 5. 训练配置与实现检查

1. 现有`zipsplat_scene_token.yaml`只是模型预设，仍会继承原入口的混合数据、450K步、batch384和RE10K/DL3DV benchmark；不能直接作为短期微调启动配置。
2. 新参数`backbone.backbone.scene_tokens`会匹配原`backbone: 0.1`学习率规则。在基准lr=3e-4时实际是3e-5；新颜色投影则是3e-4。已用现有分组函数复现。P3需显式设置新槽位的参数分组，决定热身与联合阶段各自学习率。
3. 原bounded sampler只限定target在context跨度内，没有排除中间context。固定seed42、300帧、16context/4target的1000次CPU采样，591次出现交叉；这是该设置下的诊断，不是全数据统计。若按开发计划要求互斥持出监督，需增加排除/不足帧处理；现有P1固定manifest的互斥性不受影响。
4. `ComposedDataset`对val使用child的`test_shard_dir`。正式配置必须指向开发验证划分，并将最终测试集留给独立benchmark，不能直接用最终测试反复选参。
5. 验证/保存目前多按epoch触发，混合数据规模会影响触发间隔；短微调需明确步数、保存周期、断点恢复、早停依据以及是否启用原benchmark。
6. P2已验证单卡梯度，尚未验证完整RGB/LPIPS/深度/Chamfer总损失、Adam状态显存、模型多卡DDP与恢复优化器状态。解冻范围改变时还需重新确认优化器参数组，避免新解冻参数未更新。

## 6. 下一步准备顺序

1. 以10342条暂定候选继续分辨率/质量筛选，冻结训练/验证/最终测试场景清单，排除上述135个官方测试ID及RE10K `train_tt`；先选少量训练场景作过拟合检查，再按验证结果扩大数据规模。
2. 整理ZipSplat专用环境约束，补齐训练/深度依赖、解决numpy约束、验证带`activated`的渲染器；保持QuerySplat环境不变。
3. 准备少量真实训练shard和teacher深度，核对帧序、K/pose、畸变、裁剪、depth有效率、尺度和context/target互斥；通过真实DataLoader读取检查。
4. 固定微调配置和新参数学习率组，完成单卡完整优化器步、小样本过拟合及断点恢复，再做模型DDP验证。
5. 用真实数据200–500完整step测吞吐与峰值显存，再确定批大小、长期步数以及全量预处理范围。此前按2–6秒/步估计仍未被验证。

## 7. 产物与检查边界

服务器报告根：`/root/multiview_compare/reports/scene_token/readiness_20260907/`；本地副本：`results/scene_token/readiness_20260907/`（已忽略，不提交数据）。包含数据审计、基础渲染/NCCL检查、采样/学习率诊断与源码hash。

工具：`tools/scene_token/audit_training_readiness.py`、`check_distributed_readiness.py`。检查不修改数据，只写报告；图像解码与torch内容为抽样检查，不能声称全量解码/ZIP CRC均通过。正式训练和数据派生尚未执行。

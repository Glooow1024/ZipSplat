# Scene token S=256：P2实现与验证

日期：2026-09-07。分支：`dev/scene_token`，基于`dev/multiview`的c674892。模型实现已部署到原服务器工作区；未训练、提交或推送。

## 已实现的结构

- 每视角共享初始化参数`[256,1536]`，展开后各视角拥有独立前向状态；正态初始化std=0.02。
- DA3 token布局为`[camera/CLS, register（如有）, scene, patch]`。图像绝对位置插值先完成，再插入scene；scene/register/camera的RoPE为零，patch位置保持原语义。
- 原patch全程保留，沿用原局部/全局注意力；camera槽位仍在索引0，Giant在block13注入。注入改为非原地拼接，避免破坏反向传播或activation checkpoint保存的张量。
- 导出19/29/39的scene特征，保持对应local/global拼接、3072→1536的prepare和39→29→19融合次序。三层几何KV均只包含VS场景槽位。
- 关闭scene模式的第二次k-means及训练时query采样/调度；输出K=V×256，G=32K。无效压缩配置明确报错。
- 新增`scene_color_query: Linear(1536,128)`，融合后的几何scene作为颜色query，原图全部128维颜色patch作为KV。保留原完整颜色CA及1664维head输入；scene索引不用于gather颜色patch。
- 新模式推理高斯head在autocast外以FP32运行，与训练实现一致；原模式执行原路径。

| V | K | 高斯数 |
|---:|---:|---:|
| 1 | 256 | 8192 |
| 2 | 512 | 16384 |
| 4 | 1024 | 32768 |
| 8 | 2048 | 65536 |
| 12 | 3072 | 98304 |
| 16 | 4096 | 131072 |

S256对应原模型约79.01%的query保留率，不等于预实验的75%档；同K原模型对照需补测，不能直接引用r0.75的指标。新增参数589,952个；其余已有参数形状保持。每视角原生序列从325变为581个token，因此本版不是DA3计算加速方案，也仍依赖完整颜色KV。

## 使用方式与checkpoint

原`ZipSplat(weights="zipsplat")`仍加载原模型。明确初始化新结构：

```python
from zipsplat import ZipSplat

predictor = ZipSplat(
    weights="zipsplat",
    scene_tokens_enabled=True,
    num_scene_tokens=256,
    scene_token_init_from_base=True,
).cuda()
core_model = predictor.model  # 可微核心；predictor.forward自身有no_grad，仅用于推理
```

这只是初始化，新增槽位/颜色投影没有学到重建能力。初始化原checkpoint时只允许缺失以下三个键，并逐项输出日志：

```text
backbone.backbone.scene_tokens
scene_color_query.weight
scene_color_query.bias
```

其他缺失、意外键、维度不匹配、只保存部分新增参数均报错。scene checkpoint加载到原结构也会拒绝。训练器原有保存格式含`conf.model`；推理`save_checkpoint(path)`保存`model_conf`。之后`ZipSplat(weights=path)`自动恢复结构，显式指定的S与checkpoint冲突时提前拒绝。完整训练checkpoint只支持与独立推理架构一致的配置，不静默忽略不同attention调度/头数等。

训练模型预设：[zipsplat_scene_token.yaml](../splatfactory/configs/model/zipsplat_scene_token.yaml)。它继承原模型配置，默认NoPrior、S256、全query、activation checkpoint；warmup冻结已有DA3参数，但scene槽位仍可训练。`freeze_backbone_except_scene=false`用于后续联合适配，原相机编码器仍按既有配置冻结。

原Hydra入口中选择`model=zipsplat_scene_token`即可启用这个模型预设。**正式训练的数据划分、数据格式、batch、学习率和步数仍需在P3固定**；不要直接启动原450k步/大batch配置。已验证Hydra预设能解析；本轮没有启动正式训练器或声称整个数据管线已就绪。

## 验证结果

1. 原真实checkpoint改动前/后，固定两视角252输入，means/scales/quats/opacities/SH最大绝对差全部0。
2. `tests/test_scene_tokens.py`六组测试通过：两套主干禁用功能与Git基线逐项相同；非方图/register/多视角布局；跨视角信息传递；多层/aux输出；FP16/BF16 checkpoint反传；颜色query/KV数量；冻结decoder后slot梯度；checkpoint误加载及往返保存。
3. 真实Giant，两套实现加载完全相同scene参数，在FP32与BF16-autocast下，五类高斯参数最大绝对差全部0，训练配置到推理配置转换通过。
4. 真实checkpoint，S256、V1/2/8/16的三层特征均为`[1,V,256,3072]`，K/G正确；warmup前后向均通过。scene、颜色投影、prepare和head梯度有限且非零。
5. 真实Giant联合解冻的V1/16前后向通过，最后DA3块QKV也有非零梯度。单次BF16参数损失冒烟中，16视角warmup峰值约10.11GiB，联合解冻约11.35GiB。**不含优化器状态、正式目标图渲染/损失、DDP开销，不能直接作为完整训练显存预算。**
6. 真实Giant两视角，经64×64高斯渲染器的RGB MSE反传通过，scene/颜色投影/prepare/head均有非零梯度。本项使用随机RGB和合成相机，仅验证可微链路；未优化参数、未评估重建质量。

脚本：[单元测试](../tests/test_scene_tokens.py)、[真实模型验证](../tools/scene_token/validate_scene_model.py)、[Giant数值对照](../tools/scene_token/check_scene_parity.py)。服务器权威记录：`/root/multiview_compare/experiments/scene_token/p2_256/`；本地副本：`results/scene_token/p2_256/`。

单次冒烟计时没有完整warmup/重复统计，不能与P1性能表比较。新参数未训练，尚无scene token重建质量或压缩收益结论。

## 环境变更与复现

全部验证使用`/root/zipsplat/ZipSplat/.venv/bin/python`。保存`environment_before.txt`与`environment_after.txt`；新增训练模型导入所需的plotly、omegaconf、hydra-core、scikit-learn、tensordict及其依赖，编译安装chamferdist和fused-ssim；未升级既有torch/numpy等包。

现有torch为2.13.0/CUDA13，系统nvcc为12.9。通过既有`cuda-toolkit==13.0.3.0`的`nvcc,cccl` extras在本虚拟环境新增13.0编译组件，未改系统CUDA。编译使用：

```bash
CUDA_HOME=/root/zipsplat/ZipSplat/.venv/lib/python3.12/site-packages/nvidia/cu13
MAX_JOBS=2
TORCH_CUDA_ARCH_LIST=9.0
```

虚拟环境该CUDA的lib目录缺少未版本化链接，新增`libcudart.so -> libcudart.so.13`供链接器使用。chamferdist=1.0.3；fused-ssim来自README指定仓库rahul-goel/fused-ssim，commit a7c48d6dd7ac6dc39a7958c7c4452e0b10418f38。详细版本以环境快照为准。其他完整训练依赖和数据入口仍在P3启动前核验。

## 下一阶段

P2接口实现和验证完成。P3先冻结独立训练/验证/测试场景清单，准备数据入口、训练超参及优化器显存预算，再做小样本过拟合与联合适配。用户指定S256优先；同K基线须补测约79.01%保留率，不把75%结果当成严格匹配。

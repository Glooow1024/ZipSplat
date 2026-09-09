# k-means 预实验执行口径（2026-09-07）

执行分支：`dev/scene_token`。本轮用户已授权运行预实验，不实现或训练场景token。

## 实际部署与版本

- 本地单一工作区 `F:/Development/multiview/ZipSplat`；远端单一工作区 `/root/zipsplat/ZipSplat`，从远端dev/multiview的c674892创建dev/scene_token。模型源码未修改。
- 本地开发计划已有用户修改，保留其内容。本次只新增实验工具、口径文档与进度；未提交或推送。
- 权重 `/root/.cache/torch/hub/zipsplat/zipsplat-da3g-252p.tar`。环境 `/root/zipsplat/ZipSplat/.venv/bin/python`，Torch 2.13.0+cu130、H100；实际模型文件从上述仓库导入。
- `20260907_v1`为已完成的01/8views两比例冒烟；`20260907_v2`使用相同输入清单，补充稳定阶段计时与内存口径。两个版本不合并性能结果。
- 每个v2结果携带配置签名；签名包含manifest、checkpoint、模型Python源码和runner的SHA256，以及精度/步骤/计时参数。改变签名不得静默复用旧结果。

## 数据与配准

- 输入沿用short60的4/8/12/16视角清单，所有context并集为28帧。固定8个target为frame_00002、00008、00016、00024、00037、00045、00053、00059；与所有context集合均不相交。
- 9场景有效；06在前60帧缺context GT，显式排除。每场景使用同一套完整COLMAP或transforms轨迹，不能混合两个坐标系。manifest保存每张图片的哈希、原尺寸、252内参、OpenCV c2w、位姿来源与哈希。
- 原始图像按现有ZipSplat方式中心方形裁剪，PIL Lanczos缩至252；内参按实际源尺寸及整数裁剪偏移变换。保留原图针孔近似，不另做畸变矫正；distortion参数保存供追溯。
- 重建仅输入context RGB。所有GT先相对首context归一化；通过context RGB拟合共享Sim(3)，将同一变换用于target。
- 初始尺度在0.03到30的17个对数点粗搜，再围绕最佳尺度±1.6倍搜索11点；共享旋转/平移lr=0.01，log-scale lr=0.02，Adam120步，白背景MSE。每次评估损失对应更新前参数，保存最优状态；高斯固定。
- A：r=1拟合结果跨预算复用。B（主表）：每预算用相同预算独立拟合。没有逐视角自由位姿优化，target RGB从不参与拟合。低质量可能含共享相机模型/针孔近似误差；A也可能含预算引发的坐标漂移。
- PSNR/SSIM(skimage默认window、channel_axis=2)/LPIPS-Alex v0.1均在252×252计算；预测先clamp到[0,1]。逐帧→逐场景/视角平均；完整矩阵每场景等权。context和target、A和B独立报告，不与旧256报告混合。

## 执行与可复现检查

- 原模型FP32，关闭TF32，固定随机种子20260907，eval且权重requires_grad=False。k-means初始化linspace、5迭代，BF16 autocast设置沿用原模型内部实现；主体FP32不代表聚类实现被替换为纯FP32。
- 冒烟cached阶段前向与原模型直接前向、重复原前向的高斯参数最大绝对误差均为0；不改变模型压缩算法。
- 小规模：04/05/09×8/16视角×六保留率，共36组。选择依据来自运行前图像检查：较大平面、密集货架、室外植被/栏杆。
- 完整粗扫：9×4×6=216组，六保留率为1、1/2、1/4、1/8、1/16、1/32，复用一致的小规模结果。
- 分阶段quality pass保留冷计时，仅供审计；报告稳定性能使用3warmup+10repeats完整模型前向中位数。另对缓存后的kmeans/geometry/color/head独立热身重复计时。
- 主扫的完整模型时间从已预处理GPU输入开始，不包含磁盘、裁剪、配准和指标。主进程显存含常驻特征和LPIPS，另记录前向增量峰值，不宣称为独立部署显存。
- 独立profile在04/05、4/8/12/16视角与1/.5/.125/.03125保留率测CPU RGB→高斯，包含预处理和H2D、排除磁盘加载。单独记录各stage及不加hook的总延迟，避免同步钩子污染端到端时间。

## 路径与工具

- 正式结果 `/root/multiview_compare/experiments/scene_token/kmeans_sweep/20260907_v2/`。
- 报告 `/root/multiview_compare/reports/scene_token/20260907_v2/`。
- 本地报告副本 `results/scene_token/20260907_v2/`，已加入专用ignore，不上传数据产物。
- `tools/scene_token/audit_kmeans.py`：不可覆盖的输入/位姿manifest核验。
- `tools/scene_token/run_kmeans.py`：可续跑worker、两套配准协议、指标和稳定计时。
- `tools/scene_token/report_kmeans.py`：覆盖审计、质量/预算CSV、曲线与代表图。
- `tools/scene_token/profile_kmeans.py`：不含质量缓存的独立端到端/分阶段计时。

完成记录：粗扫后按完整B-target均值相对r=1下降0.5/1/2dB所在相邻区间取算术中点，得到0.75/0.375/0.1875三个统一比例，完成108组细扫；不按单场景最优点挑结果。固定K=324/648跨V对照复用已有同K结果，仅补充12视角的18组。合计342组质量实验与32个独立性能配置均完成，完整覆盖检查通过。最终结论见KMEANS_PREEXPERIMENT_RESULTS_ZH.md；所有逐帧数据、图像、配准参数、环境与源码哈希保存在结果目录。本轮未提交或推送代码，未修改模型权重或开始场景token训练。

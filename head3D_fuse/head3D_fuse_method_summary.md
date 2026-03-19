# head3D_fuse 模块方法总结说明文档

## 1. 总体流程与架构

`head3D_fuse` 实现多视角3D头部关键点融合、时序平滑和可视化。主流程由 `main.py`（多进程调度）和 `infer.py`（单人单环境融合）负责，支持批量处理与高效并行。

- 主入口：`main.py` 利用多进程分配任务，批量处理不同人员和环境的数据。
- 单人融合流程：`infer.py` 的 `_fuse_single_person_env` 函数负责加载、过滤、融合、平滑和保存每一帧的关键点。

---

## 2. 多视角融合算法


核心融合逻辑在 `fuse/fuse.py`，主要方法包括：

### 详细融合方法

1. **中位数/均值融合**
	- 对每个关键点，分别收集所有视角的坐标。
	- 采用中位数（median）或均值（mean）对每个坐标进行融合，减少异常值影响。
	- 适用于视角间误差较小或偶有缺失的场景。

2. **Procrustes对齐（空间变换）**
	- 先对不同视角的关键点进行空间对齐（旋转、缩放、平移），使其落到统一坐标系。
	- 通过最小化欧氏距离，计算最佳变换参数（scale、rotation、translation）。
	- 对齐后再进行融合，提升融合精度，适用于视角间存在较大空间差异的情况。

3. **视角变换**
	- 利用相机外参（旋转矩阵R、平移向量t），将各视角关键点统一到世界坐标系。
	- 支持 world_to_camera 和 camera_to_world 两种模式，保证融合结果空间一致。

---

## 3. 时序平滑


为提升时序稳定性，`smooth/temporal_smooth.py` 提供多种滤波算法：

### 详细时间平滑方法

1. **高斯滤波（Gaussian）**
	- 对每个关键点的时序轨迹，采用高斯核进行卷积平滑。
	- 参数 sigma 控制平滑程度，sigma 越大，平滑效果越强。
	- 能有效抑制高频噪声，保持整体趋势。

2. **Savgol滤波（Savgol）**
	- 使用 Savitzky-Golay 算法对序列进行多项式拟合平滑。
	- 保留更多细节，适合需要平滑但不丢失运动特征的场景。

3. **Kalman滤波（Kalman）**
	- 基于状态空间模型，结合预测与观测，动态平滑关键点轨迹。
	- 能处理缺失数据和动态变化，适合复杂运动场景。

4. **双边滤波（Bilateral）**
	- 同时考虑空间和时间相似性，平滑时保留边缘和突变。
	- 适合需要保留动作边界的场景。

---

## 4. 可视化与对比

可视化工具在 `visualization/` 文件夹：

- 骨架渲染：`skeleton_visualizer.py` 定义头部、肩部、手部等关键点及连接关系，实现3D骨架绘制。
- 帧图像合成：`vis_utils.py` 支持多视角图像与融合结果的并排展示，便于对比分析。
- 视频生成：`merge_video.py` 将帧图像合成视频，便于结果回顾与展示。

---

## 5. 质量评估与对比分析

- 一致性评估：`fuse/compare_fused.py` 计算融合结果与各视角的欧氏距离、时序抖动（jitter）等指标，分析融合效果。
- 平滑效果对比：`smooth/compare_fused_smoothed.py` 评估平滑前后关键点的误差变化。

---

## 6. 主要技术与创新点

- 多视角3D关键点融合（中位数/均值/Procrustes对齐）
- 时序滤波（高斯、Savgol、Kalman、双边）
- 骨架结构定义与可视化
- 多进程批量处理架构
- 质量评估与对比分析

---

### 参考代码片段（部分核心实现）

- 多视角融合与对齐：[fuse/fuse.py](code/head3D_fuse/fuse/fuse.py#L47-L111)、[fuse/fuse.py](code/head3D_fuse/fuse/fuse.py#L133-L214)
- 时序平滑：[smooth/temporal_smooth.py](code/head3D_fuse/smooth/temporal_smooth.py#L44-L98)
- 骨架渲染：[visualization/skeleton_visualizer.py](code/head3D_fuse/visualization/skeleton_visualizer.py#L27-L85)
- 视频合成：[visualization/merge_video.py](code/head3D_fuse/visualization/merge_video.py#L15-L55)
- 质量评估：[fuse/compare_fused.py](code/head3D_fuse/fuse/compare_fused.py#L14-L60)
- 主流程调度：[main.py](code/head3D_fuse/main.py#L112-L192)

---

如需更详细的算法描述或代码注释，可进一步提取具体函数实现与参数说明。此文档可直接用于论文方法部分，涵盖核心流程、算法、创新点与评估方式。

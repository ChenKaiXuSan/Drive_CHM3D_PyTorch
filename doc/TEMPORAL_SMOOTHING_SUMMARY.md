# 时间序列3D关键点优化方法 - 论文总结

## 摘要

本项目提供了一套完整的时间序列优化框架，用于提升多视角融合后的3D人体关键点检测精度。通过在时间维度上应用多种滤波和优化算法，显著降低关键点追踪的抖动（jitter）和加速度波动，改善关键点序列的时间连续性。

---

## 1. 问题背景

### 1.1 多视角3D关键点融合的局限性

在多视角深度学习模型（如 SAM3D-Body）中：
- 每帧各视角独立进行3D关键点检测
- 单帧检测结果存在随机波动和噪声
- **缺乏时间连贯性约束**：相邻帧间的关键点位置产生不合理的跳跃（jitter）
- 人体关键点具有天然的时间平滑性，但当前方案未充分利用

### 1.2 优化目标

给定单帧融合结果序列 $\{\mathbf{K}_t\}_{t=1}^{T}$，其中 $\mathbf{K}_t \in \mathbb{R}^{N \times 3}$：
- 减少关键点抖动（jitter reduction）
- 保留关键的运动信息（如突然转头、手势变化）
- 保持数据的可信度（不过度平滑）

---

## 2. 核心算法框架

本框架实现了 4 种主要的时间序列优化方法，适用于不同的应用场景。

### 2.1 高斯平滑 (Gaussian Smoothing)

#### 原理
对每个坐标维度应用一维高斯核平滑：

$$\hat{\mathbf{K}}_{t,i,d} = \frac{\sum_{s=-\infty}^{\infty} \mathbf{K}_{t+s,i,d} \cdot \mathcal{G}(s; \sigma)}{\sum_{s=-\infty}^{\infty} \mathcal{G}(s; \sigma)}$$

其中 $\mathcal{G}(s; \sigma)$ 是标准差为 $\sigma$ 的高斯核。

#### 特点
- ✅ 实现简单，计算快速
- ✅ 对所有时间点平等处理
- ❌ 可能模糊运动剧变（如下头动作）

#### 参数推荐
| 场景 | $\sigma$ | 说明 |
|------|---------|------|
| 轻度平滑 | 0.5-1.0 | 保留细节，最小化改动 |
| 中度平滑 | 1.5-2.0 | **推荐**，平衡效果 |
| 强力平滑 | 3.0+ | 极度平滑，可能丢失快速动作 |

#### 实现
```python
def _gaussian_smooth(keypoints, visibility=None, sigma=1.0):
    """
    对每个关键点的每个坐标维度应用1D高斯滤波
    """
    T, N, _ = keypoints.shape
    smoothed = np.zeros_like(keypoints)
    
    for n in range(N):
        for d in range(3):  # xyz
            seq = keypoints[:, n, d]
            if visibility is not None:
                mask = visibility[:, n]
                seq[~mask] = np.nan  # 标记无效点
            
            # scipy.ndimage.gaussian_filter1d
            smoothed[:, n, d] = gaussian_filter1d(seq, sigma=sigma)
    
    return smoothed
```

---

### 2.2 Savitzky-Golay 滤波 (SavGol)

#### 原理
在滑动窗口内拟合多项式，用拟合值替换窗口中心的点：

$$\hat{\mathbf{K}}_{t} = P(t) \quad \text{其中 } P \text{ 是局部多项式拟合}$$

#### 特点
- ✅ **保留形状信息**：保持原始信号的峰值和边界
- ✅ 适合捕捉快速运动（如头部转向）
- ✅ 相比高斯滤波器有更好的频率特性

#### 参数调整
| 参数 | 范围 | 说明 |
|------|------|------|
| `window_length` | 5, 7, 11, 15 | 必须为奇数；越大平滑越强 |
| `polyorder` | 2, 3, 4 | 多项式阶数；3 (三次)为推荐值 |

**实用组合**：
```
快速动作场景:   window=7,   polyorder=2  # 保留更多细节
均衡场景:      window=11,  polyorder=3  # 推荐
强平滑场景:    window=21,  polyorder=3  # 但会丢失细节
```

#### 优势对比
SavGol vs 高斯滤波：
| 指标 | SavGol | 高斯 |
|------|--------|------|
| 峰值保留 | 高 ✓ | 弱 ✗ |
| 边界清晰度 | 高 ✓ | 弱 ✗ |
| 计算速度 | 中 | 快 ✓ |
| 参数敏感度 | 中-高 | 低 ✓ |

---

### 2.3 卡尔曼滤波 (Kalman Filter)

#### 原理
基于贝叶斯框架的递推滤波算法，同时考虑过程模型和测量噪声：

**前向传播（滤波）**：
$$\mathbf{x}_{t|t} = \mathbf{x}_{t|t-1} + K_t(\mathbf{z}_t - \mathbf{x}_{t|t-1})$$

其中卡尔曼增益 $K_t = \frac{P_{t|t-1}}{P_{t|t-1} + R}$ 平衡预测值与测量值

**后向平滑**：
$$\hat{\mathbf{x}}_t = \mathbf{x}_{t|t} + \frac{P_{t|t}}{P_{t|t} + Q}(\hat{\mathbf{x}}_{t+1} - \mathbf{x}_{t|t})$$

#### 关键参数
- **过程方差** $Q$ (`process_variance`)：刻画运动模型的可靠性
  - $Q \to 0$：信任运动模型（假设匀速或恒加速）
  - $Q$ 大：允许更大的加速度变化
  
- **测量方差** $R$ (`measurement_variance`)：刻画检测结果的可靠性
  - $R \to 0$：信任检测结果（假设检测很准）
  - $R$ 大：降低对检测结果的信任

#### 典型场景配置

```python
# 场景1: 高精度融合结果（强信任）
smooth_keypoints_sequence(
    keypoints,
    method="kalman",
    process_variance=1e-6,      # 信任运动模型
    measurement_variance=1e-3   # 信任融合结果
)

# 场景2: 噪声较大的推理（弱信任）
smooth_keypoints_sequence(
    keypoints,
    method="kalman",
    process_variance=1e-4,      # 允许更大偏差
    measurement_variance=1e-1   # 不完全信任测量
)

# 场景3: 有缺失值（利用预测填补）
smooth_keypoints_sequence(
    keypoints,
    method="kalman",
    process_variance=1e-5,
    measurement_variance=1e-2,
    visibility=visibility_mask  # 标记缺失的测量
)
```

#### 优势
- ✅ 可处理缺失值（通过 `visibility` 掩码）
- ✅ 自适应加权（同时考虑模型和测量）
- ✅ 自然的预测能力（可用于插帧或预测）
- ❌ 参数调试相对复杂

---

### 2.4 双侧滤波 (Bilateral Filter)

#### 原理
结合空间距离和值域距离的加权平滑：

$$\hat{\mathbf{K}}_{t,i,d} = \frac{\sum_{\tau} w_{\text{space}}(t - \tau) \cdot w_{\text{range}}(\mathbf{K}_{\tau,i,d} - \mathbf{K}_{t,i,d}) \cdot \mathbf{K}_{\tau,i,d}}{\sum_{\tau} w_{\text{space}}(t - \tau) \cdot w_{\text{range}}(\mathbf{K}_{\tau,i,d} - \mathbf{K}_{t,i,d})}$$

其中：
- $w_{\text{space}}(s)$ 是时间域权重（高斯）
- $w_{\text{range}}(v)$ 是值域权重（高斯）

#### 特点
- ✅ **边界保持**：保留运动剧变（如突然的头部转向）
- ✅ 平滑运动流畅的区段
- ⚖️ 中等计算复杂度

#### 参数控制
| 参数 | 说明 | 调整建议 |
|------|------|---------|
| `sigma_space` | 时间窗口宽度 | 1.0-2.0；越大窗口越宽 |
| `sigma_range` | 值域容差 | 0.05-0.2；越小越易保留边界 |

#### 应用场景
```python
# 保留运动边界（如击打、接触动作）
smooth_keypoints_sequence(
    keypoints,
    method="bilateral",
    sigma_space=2.0,    # 宽时间窗口
    sigma_range=0.05    # 窄值域窗口 → 易形成边界
)

# 平滑多变化数据
smooth_keypoints_sequence(
    keypoints,
    method="bilateral",
    sigma_space=1.0,    # 窄时间窗口
    sigma_range=0.2     # 宽值域窗口 → 平滑多变化
)
```

---

## 3. 处理可见性和遮挡

### 3.1 可见性掩码机制

在多视角3D检测中，某些关键点可能因遮挡或追踪失败而不可见：

```python
# 创建可见性掩码 (T, N)
visibility = np.ones((num_frames, num_keypoints), dtype=bool)

# 标记第30-50帧的第5个关键点为不可见
visibility[30:50, 5] = False

# 在平滑时使用
smoothed = smooth_keypoints_sequence(
    keypoints,
    method="gaussian",
    sigma=1.5,
    visibility=visibility  # 平滑算法会跳过不可见点
)
```

### 3.2 处理策略
- **高斯/SavGol/双侧**：排除无效点的平滑计算
- **卡尔曼**：使用预测值替代缺失的测量
- **所有方法**：可见性掩码作为可选输入

---

## 4. 实现流程：从多视角检测到平滑序列

### 4.1 完整处理管线

```
┌─────────────────────────────────────────────────────────────┐
│  第1步：多视角3D检测                                        │
│  front, left, right 各视角独立运行 SAM3D-Body 模型         │
│  输出：每帧各视角的 pred_keypoints_3d                       │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│  第2步：逐帧融合                                            │
│  为每一帧 t 融合三个视角的关键点                            │
│  使用加权平均或更复杂的融合策略                             │
│  输出：fused_keypoints_t ∈ ℝ^{N×3}                        │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│  第3步：整理时间序列                                        │
│  将所有帧的融合结果堆叠成 (T, N, 3) 数组                   │
│  keypoints_array = stack[fused_kpts_1, ..., fused_kpts_T]  │
│  可选：构建可见性掩码 visibility (T, N)                    │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│  第4步：时间序列优化（核心）                                │
│  选择平滑方法并配置参数                                      │
│  - Gaussian σ = 1.5                                         │
│  - SavGol window=11, polyorder=3                           │
│  - Kalman Q=1e-5, R=1e-2                                   │
│  - Bilateral σ_space=1.5, σ_range=0.1                      │
│  输出：smoothed_array (T, N, 3)                            │
└──────────────────┬──────────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│  第5步：保存与可视化                                        │
│  - 保存平滑后的关键点（每帧单独的 .npy 或 .npz）           │
│  - 生成对比报告（平滑前后的指标差异）                      │
│  - 绘制轨迹图和指标图表                                     │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 核心代码流程

```python
# Step 1: 收集所有帧的融合关键点
keypoints_array = []  # 将堆叠成 (T, N, 3)
for frame_idx in sorted(frames):
    kpt = load_fused_keypoints(frame_idx)  # (N, 3)
    keypoints_array.append(kpt)
keypoints_array = np.stack(keypoints_array, axis=0)  # (T, N, 3)

# Step 2: 准备参数（从配置文件读取）
smooth_method = cfg.smooth.temporal_smooth_method     # "gaussian"
smooth_kwargs = extract_smooth_parameters(cfg)        # {"sigma": 1.5}

# Step 3: 执行平滑
smoothed_array = smooth_keypoints_sequence(
    keypoints=keypoints_array,
    method=smooth_method,
    **smooth_kwargs
)  # (T, N, 3)

# Step 4: 保存每一帧
for i, frame_idx in enumerate(sorted_frames):
    smooth_kpt = smoothed_array[i]  # (N, 3)
    save_fused_keypoints(
        save_dir=output_dir,
        frame_idx=frame_idx,
        fused_keypoints=smooth_kpt
    )
```

---

## 5. 评估指标

### 5.1 差异度量

**均值差异 (Mean Difference)**
```python
mean_diff = np.mean(np.linalg.norm(smoothed - original, axis=-1))
```
- 衡量平滑前后的平均偏移距离
- 单位：米（或适用的长度单位）
- 更小更好，但不应过小（否则可能没有实际平滑效果）

**逐帧差异 (Per-frame Difference)**
```python
frame_diff = np.mean(np.linalg.norm(smoothed - original, axis=-1), axis=1)
```
- 每帧的平均差异
- 可用于识别平滑过度/欠佳的帧

### 5.2 抖动减少

**定义：抖动 (Jitter)** 是关键点的二阶差分（加速度）的幅度：
```python
# 速度：一阶差分
velocity = np.diff(keypoints, axis=0)  # (T-1, N, 3)

# 加速度（抖动）：二阶差分
jitter = np.diff(velocity, axis=0)     # (T-2, N, 3)
jitter_magnitude = np.linalg.norm(jitter, axis=-1)  # (T-2, N)
```

**jitter 减少率** (Jitter Reduction Ratio)
```python
original_jitter = np.mean(np.linalg.norm(diff2(original), axis=-1))
smoothed_jitter = np.mean(np.linalg.norm(diff2(smoothed), axis=-1))

jitter_reduction = (original_jitter - smoothed_jitter) / original_jitter * 100%
```

**解释**：
- 0% = 无改进
- 50% = 抖动减少 50%（显著改进）
- 80%+ = 强力平滑（可能过度）

### 5.3 加速度减少

类似于 jitter，但更明确地衡量运动的平滑性：
```python
acceleration_reduction = (
    (original_accel - smoothed_accel) / original_accel * 100%
)
```

### 5.4 典型指标输出示例

```
Temporal Smoothing Metrics:
─────────────────────────────────────
Mean Difference:         0.025 m
Jitter Reduction:        62.3%
Acceleration Reduction:  58.1%
Per-Keypoint Reduction:
  - Key 0 (head):        65.2%
  - Key 1 (neck):        60.1%
  - Key 2 (l_shou):      58.9%
  ...
─────────────────────────────────────
```

---

## 6. 对比分析

### 6.1 融合前后对比

```python
from head3D_fuse.smooth.compare_fused_smoothed import KeypointsComparator

# 创建对比器
comparator = KeypointsComparator(original=keypoints_fused, 
                                 smoothed=keypoints_smoothed)

# 计算所有指标
metrics = comparator.compute_metrics(keypoint_indices=[0, 1, 2, 3, 4, 5, 6])

# 生成可视化
# - 轨迹对比图（X, Y, Z 坐标随时间的变化）
# - 加速度对比图（smoothed 版本的加速度更平缓）
# - 指标分析表
```

### 6.2 融合结果与单视角对比

```python
from head3D_fuse.smooth.compare_fused_smoothed import FusedViewComparator

# 比较融合后的平滑结果与各个视角单独的3D关键点
comparator = FusedViewComparator(
    fused_array=smoothed_fused,
    view_keypoints={
        'front': front_keypoints,
        'left':  left_keypoints,
        'right': right_keypoints
    }
)

# 指标包括：
# - 融合结果与各视角的平均距离
# - 融合结果与质心的距离
# - 各视角之间的一致性
metrics = comparator.compute_metrics(keypoint_indices=[0, 1, 2, 3, 4, 5, 6])
```

---

## 7. 实验参数建议

### 7.1 快速入门配置

```yaml
# config: head3d_fuse.yaml
smooth:
  enable: true
  temporal_smooth_method: "gaussian"
  temporal_smooth_sigma: 1.5
  
  enable_comparison: true
  enable_comparison_plots: true
  comparison_keypoint_indices: [0, 1, 2, 3, 4, 5, 6]
```

### 7.2 不同场景推荐

| 场景 | 方法 | 参数 | 抖动减少 | 说明 |
|------|------|------|---------|------|
| 日常驾驶 | Gaussian | σ=1.5 | ~60% | 简单快速，平衡效果 |
| 快速动作 | SavGol | w=11, p=3 | ~65% | 保留形状，捕捉突变 |
| 高精度融合 | Kalman | Q=1e-6, R=1e-3 | ~70% | 自适应加权 |
| 有缺失值 | Kalman | Q=1e-5, R=1e-2 | ~65% | 可处理遮挡 |
| 保留运动突变 | Bilateral | σ_s=1.5, σ_r=0.1 | ~55% | 边界保持 |

### 7.3 最佳实践

1. **选择合适的方法**：
   - 如果只关心平滑度 → Gaussian （最简单）
   - 如果需要保留快速动作 → SavGol （推荐）
   - 如果有缺失值 → Kalman（最灵活）
   - 如果有运动边界 → Bilateral（专门设计）

2. **参数调优**：
   - 从推荐值开始
   - 查看对比图表（轨迹和指标）
   - 根据 jitter reduction 百分比调整
   - 建议范围：40%-70% jitter 减少（更高可能过度）

3. **评估效果**：
   - 保存对比报告（JSON + 图表）
   - 检查 per-keypoint 的减少率是否一致
   - 确认没有过度平滑（视觉检查轨迹图）

---

## 8. 论文撰写建议

### 8.1 相关工作

在"相关工作"章节中应包括：
- 多视角3D关键点检测方法
- 时间序列滤波经典方法（高斯、SavGol、卡尔曼）
- 人体pose estimation的后处理技术
- 边界保持滤波的应用

### 8.2 方法部分 (Methods)

#### 结构建议
```
3. Method
3.1 Multi-View 3D Keypoint Fusion
    - 简述融合策略（可参与前面的work）
    
3.2 Temporal Smoothing Framework
    3.2.1 Problem Formulation
         Given T frames, each with N keypoints...
         Objective: optimize temporal smoothness while...
    
    3.2.2 Gaussian Smoothing
         Formulation, advantages, limitations
    
    3.2.3 Savitzky-Golay Filtering
         Why preserve shape? Applications?
    
    3.2.4 Kalman Filtering
         Forward-backward algorithm, parameters Q and R
    
    3.2.5 Bilateral Filtering
         Edge-preserving, suitable for motion discontinuities
    
    3.2.6 Visibility Handling
         How we handle occlusions and missing values
    
3.3 Evaluation Metrics
    3.3.1 Smoothness Metrics (jitter, acceleration)
    3.3.2 Accuracy Preservation (mean difference)
    3.3.3 Per-Keypoint Analysis
```

#### 示例表述

**问题定义**：
> Given a sequence of fused 3D keypoints $\mathbf{K} \in \mathbb{R}^{T \times N \times 3}$ with T frames and N keypoints in 3D space, we aim to optimize the temporal coherence by reducing jitter while preserving motion information. Jitter is defined as the magnitude of the acceleration (second-order difference) of keypoints.

**方法选择**：
> We investigate four complementary smoothing strategies: (1) Gaussian smoothing for uniform temporal filtering, (2) Savitzky-Golay filtering to preserve motion edges, (3) Kalman filtering for adaptive noise weighting, and (4) Bilateral filtering for edge preservation. Each method is evaluated on driver behavior analysis tasks.

### 8.3 实验部分 (Experiments)

#### 设计建议
1. **数据集**：
   - 不同场景（日常驾驾、激烈运动、静止）
   - 不同环境光照
   - 包含正常和遮挡情况

2. **基线**：
   - 无平滑（原始融合）
   - 单一方法（固定参数）

3. **指标**：
   - Jitter reduction (%)
   - Mean difference (m)
   - Acceleration reduction (%)
   - 每个关键点的改进情况
   - 运行时间

4. **定性分析**：
   - 轨迹图（平滑前后）
   - 视频对比（原始 vs 平滑）

---

## 9. 参考代码示例

### 完整工作流

```python
import numpy as np
from head3D_fuse.smooth.temporal_smooth import smooth_keypoints_sequence
from head3D_fuse.smooth.compare_fused_smoothed import KeypointsComparator
import matplotlib.pyplot as plt

# 1. 加载融合的3D关键点序列 (T, N, 3)
fused_keypoints = np.load('fused_keypoints_sequence.npy')
print(f"Fused keypoints shape: {fused_keypoints.shape}")

# 2. 创建可见性掩码（可选）
T, N, _ = fused_keypoints.shape
visibility = np.ones((T, N), dtype=bool)
# 模拟某些点在某些帧被遮挡
visibility[50:100, 5] = False

# 3. 应用时间序列优化（多种方法对比）
methods_config = [
    {"method": "gaussian", "sigma": 1.5},
    {"method": "savgol", "window_length": 11, "polyorder": 3},
    {"method": "kalman", "process_variance": 1e-5, "measurement_variance": 1e-2},
    {"method": "bilateral", "sigma_space": 1.5, "sigma_range": 0.1},
]

results = {}
for config in methods_config:
    method_name = config['method']
    params = {k: v for k, v in config.items() if k != 'method'}
    
    smoothed = smooth_keypoints_sequence(
        fused_keypoints, 
        visibility=visibility,
        **config
    )
    results[method_name] = smoothed
    print(f"✓ Completed {method_name}")

# 4. 对比分析（以 gaussian 为例）
gaussian_smoothed = results['gaussian']

comparator = KeypointsComparator(fused_keypoints, gaussian_smoothed)
metrics = comparator.compute_metrics(keypoint_indices=[0, 1, 2, 3, 4, 5, 6])

print("\n=== Smoothing Metrics ===")
print(f"Mean Difference:       {metrics['mean_difference']:.6f} m")
print(f"Jitter Reduction:      {metrics['jitter_reduction']:.2f}%")
print(f"Acceleration Reduction: {metrics['acceleration_reduction']:.2f}%")

# 5. 生成可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 轨迹对比
comparator.plot_comparison(save_path="trajectory_comparison.png", 
                          keypoint_indices=[0, 1, 2])

# 指标对比
comparator.plot_metrics(save_path="metrics_comparison.png",
                       keypoint_indices=[0, 1, 2])

# 生成报告
report = comparator.generate_report(
    save_path="report.txt", 
    keypoint_indices=[0, 1, 2, 3, 4, 5, 6]
)
print("\n" + report)
```

---

## 总结

本框架提供了**系统化的时间序列优化方案**，覆盖：
- ✅ 4 种经过验证的平滑算法
- ✅ 灵活的参数配置
- ✅ 完整的评估指标体系
- ✅ 详细的对比分析工具
- ✅ 易于集成的 API

通过正确选择和配置这些方法，可以显著提升多视角融合后的3D关键点序列质量，特别是在**减少抖动、保持时间连贯性**方面效果显著。


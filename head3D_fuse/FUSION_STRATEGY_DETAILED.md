# head3D_fuse 多视角融合策略详细说明

## 1. 融合策略总体框架

当前 head3D_fuse 模块采用**三阶段融合**流程：

```
输入多视角数据
    ↓
[阶段1] 关键点预处理与有效性检查
    ↓
[阶段2] 坐标对齐（可选）
    ↓
[阶段3] 多视角聚合
    ↓
输出融合结果
```

---

## 2. 阶段1：关键点预处理与有效性检查

### 2.1 输入读取与归一化

触发位置：[infer.py](infer.py) 中 `_normalize_keypoints` 函数

**处理步骤：**

1. 读取各视角的 `pred_keypoints_3d`（从 npz 加载）
2. **处理 batch 维度**
   - 若输入形状为 `(batch, N, 3)` 且 batch >= 1，取第 0 个
   - 最终输出形状 `(N, 3)`

3. **关键点索引过滤**
   - 定义保留索引：头部 + 肩部 + 双手
   - 代码见 KEEP_KEYPOINT_INDICES：
     ```
     0-4: 鼻子、左眼、右眼、左耳、右耳（头部）
     5-6: 左肩、右肩（肩部）
     21-62: 右手(21-41)、左手(42-62)（双手）
     67-69: 左肩峰、右肩峰、颈部（额外）
     ```
   - 若关键点个数不足保留索引，填充 NaN

4. **输出**
   - 若输入为 None，输出填充全 NaN 的数组
   - 否则输出过滤后的关键点 `(M, 3)`，M ≤ N

### 2.2 有效性判断

触发位置：[fuse.py](fuse/fuse.py) 中 `_valid_keypoints_mask` 函数

**判断条件：**

```python
finite = np.isfinite(keypoints).all(axis=-1)  # 不含 NaN/Inf
nonzero = np.linalg.norm(keypoints, axis=-1) >= zero_eps  # 模长 >= 1e-6
valid = finite & nonzero
```

**在实际融合中的应用：**

- 逐关键点计算各视角的有效掩码
- 堆叠后得到 `valid` 形状 `(num_views, num_keypoints)`
- 若某帧的 `valid` 中有任一关键点所有视角都无效，该关键点在融合结果中填 NaN

### 2.3 整帧有效性检查

触发位置：[infer.py](infer.py) 中 `_fuse_single_person_env` 函数

**检查逻辑：**

```python
missing_views = [
    view for view, kpt in keypoints_by_view.items() 
    if np.all(np.isnan(kpt))  # 整个视角全 NaN
]
if missing_views:
    logger.warning("Missing views for frame %s", frame_idx)
    continue  # 跳过该帧
```

**结论**：如果任一视角该帧全 NaN，整帧跳过不融合。

---

## 3. 阶段2：坐标对齐（可选）

融合前可选地将各视角对齐到统一坐标系。有三种模式：

### 3.1 模式1：无对齐（alignment_method = "none"）

**配置：**
```yaml
fuse:
  alignment_method: none
```

**效果**：直接使用各视角的相机坐标系，不做对齐。

### 3.2 模式2：相机外参对齐（view_transforms）

**配置：**
```yaml
fuse:
  view_transforms:  # 字典，每个视角对应一个变换
    front:
      R: [[r11, r12, r13], [...]]  # 3x3 旋转矩阵
      t: [tx, ty, tz]              # 3 平移向量
      # 或 t_wc / C 等
  transform_mode: world_to_camera  # 或 camera_to_world
```

**变换函数**：[fuse.py](fuse/fuse.py) 中 `_apply_view_transform`

**两种模式：**

1. `world_to_camera` 模式
   - 假设已知 world -> camera 变换
   - 公式：`X_world = (X_cam - t_wc) @ R` 或 `X_world = X_cam @ R + C`
   - 将各视角坐标变换到世界坐标系

2. `camera_to_world` 模式
   - 假设已知 camera -> world 变换
   - 公式：`X_world = X_cam @ R.T + t`
   - 直接投影到世界坐标系

**效果**：所有视角尽可能投影到同一坐标系，便于融合。

### 3.3 模式3a：Procrustes 对齐（alignment_method = "procrustes"）

**触发条件**：无 view_transforms 但指定 alignment_method = procrustes

**算法**：[fuse.py](fuse/fuse.py) 中 `_estimate_similarity_transform` 和 `_align_keypoints_to_reference`

**步骤：**

1. 选择参考视角（默认 view_list[0]，通常是 front）
2. 对其他每个视角，计算到参考视角的相似变换
3. 相似变换估计：
   - 最小化 `||source_aligned - target||²`
   - 包含：旋转矩阵 R (3x3)、尺度 s、平移向量 t (3)
   - 使用 SVD 求解最优旋转

**伪代码：**

```python
def _estimate_similarity_transform(source, target, allow_scale=True):
    # 中心化
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    
    # SVD 求最优旋转
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 修正行列式（确保旋转不是反射）
    if det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    
    # 尺度估计
    scale = sum(S) / sum(source_centered**2) if allow_scale else 1.0
    
    # 平移估计
    translation = target.mean() - scale * (source.mean() @ R)
    
    return scale, R, translation
```

**配置：**

```yaml
fuse:
  alignment_method: procrustes
  alignment_reference: front     # 参考视角
  alignment_scale: true          # 是否允许尺度变换
```

**优点**：无需相机外参，自动对齐多视角数据到共同坐标系。

### 3.3b 模式3b：Robust Procrustes（alignment_method = "procrustes_trimmed"）

**算法**：[fuse.py](fuse/fuse.py) 中 `_align_keypoints_trimmed`

**与 Procrustes 的区别**：

1. 迭代剔除离群点
2. 每次迭代：
   - 根据当前变换计算残差
   - 按择优比例（trim_ratio）保留最小误差的点
   - 重新估计变换
3. 直到收敛或达到最大迭代数

**适用场景**：视角间数据噪声大或包含异常值

**配置：**

```yaml
fuse:
  alignment_method: procrustes_trimmed
  alignment_trim_ratio: 0.2              # 剔除比例（0-1）
  alignment_scale: true
  alignment_max_iters: 3                 # 最多迭代次数
```

---

## 4. 阶段3：多视角聚合

### 4.1 堆叠与有效性掩码

```python
# 将各视角关键点堆叠
stacked = np.stack([kpts[view] for view in view_list], axis=0)
# 形状：(num_views, num_keypoints, 3)

# 计算有效性掩码
finite = np.isfinite(stacked).all(axis=-1)  # (num_views, num_keypoints)
nonzero = np.linalg.norm(stacked, axis=-1) >= zero_eps
valid = finite & nonzero
```

### 4.2 融合方法

配置参数：
```yaml
fuse:
  fuse_method: median  # 或 mean / first
```

#### 方法1：中位数（median）

```python
if method == "median":
    # 对每个关键点的各视角坐标取中位数
    stacked[~valid] = np.nan  # 无效值标记为 NaN
    fused_keypoint = np.nanmedian(stacked[:, keypoint_idx, :], axis=0)
```

**特点**：
- 非常鲁棒，消除单视角异常值影响
- 推荐用于三视角融合
- 当某视角两个维度有异常时仍能保持整体质量

**例子**：
```
front:  [1.0, 2.0, 3.0]
left:   [1.1, 2.1, 3.1]    <- 质量最好
right:  [5.0, 2.0, 3.0]    <- X 坐标异常

中位数结果: [1.1, 2.0, 3.0]
```

#### 方法2：均值（mean）

```python
if method == "mean":
    stacked[~valid] = np.nan
    fused_keypoint = np.nanmean(stacked[:, keypoint_idx, :], axis=0)
```

**特点**：
- 比中位数敏感
- 适合视角数据噪声小且均匀分布

#### 方法3：优先级（first）

```python
if method == "first":
    # 按视角顺序取第一个有效值
    for view_idx in range(num_views):
        if valid[view_idx, keypoint_idx]:
            fused = stacked[view_idx, keypoint_idx]
            break
```

**特点**：
- 完全依赖视角顺序
- 用于某视角数据确定优先的场景

### 4.3 输出结果

```python
fused_mask = valid.any(axis=0)         # (num_keypoints,) 该关键点至少一个视角有效
n_valid = valid.sum(axis=0)            # (num_keypoints,) 有多少视角该关键点有效

# 最终融合形状
fused: (num_keypoints, 3)              # 融合的关键点坐标
fused_mask: (num_keypoints,)           # 关键点有效掩码
n_valid: (num_keypoints,)              # 每关键点的有效视角数
```

---

## 5. 完整融合流程（代码视角）

```python
def fuse_3view_keypoints(
    keypoints_by_view,                          # {view: (N,3)} 多视角关键点
    method="median",                            # 融合方法
    zero_eps=1e-6,                              # 有效性判断阈值
    view_transforms=None,                       # 可选：相机外参
    transform_mode="world_to_camera",
    alignment_method="none",                    # none/procrustes/procrustes_trimmed
    alignment_reference=None,
    alignment_scale=True,
    alignment_trim_ratio=0.2,
    alignment_max_iters=3,
):
    # 1. 应用相机外参对齐（可选）
    if view_transforms:
        keypoints_by_view = {
            view: _apply_view_transform(kpts, view_transforms[view], transform_mode)
            for view, kpts in keypoints_by_view.items()
        }
    
    # 2. 应用 Procrustes 对齐（可选）
    elif alignment_method in ("procrustes", "procrustes_trimmed"):
        reference = keypoints_by_view[alignment_reference or view_list[0]]
        keypoints_by_view = {
            view: (reference if view == ref else align(reference, kpts, method))
            for view, kpts in keypoints_by_view.items()
        }
    
    # 3. 堆叠并计算有效掩码
    stacked = np.stack(list(keypoints_by_view.values()), axis=0)
    valid = _compute_validity(stacked, zero_eps)
    
    # 4. 融合
    if method == "median":
        fused = np.nanmedian(valid_masked_stack, axis=0)
    elif method == "mean":
        fused = np.nanmean(valid_masked_stack, axis=0)
    
    return fused, fused_mask, n_valid
```

---

## 6. 融合流程总结表

| 阶段 | 功能 | 关键函数 | 输入 | 输出 |
|------|------|---------|------|------|
| 预处理 | 读取、批次处理、索引过滤、NaN 检查 | `_normalize_keypoints` | npz 关键点 | (M, 3) 无效值已标记 |
| 有效性检查 | 计算每个关键点每视角的有效掩码 | `_valid_keypoints_mask` | (N, 3) | (num_keypoints,) bool |
| 帧级检查 | 检查整视角是否全 NaN，决定是否跳过 | infer.py 逻辑 | 预处理结果 | yes/no 继续 |
| 对齐（可选1） | 通过相机外参投影到世界坐标系 | `_apply_view_transform` | 相机坐标 + 外参 | 世界坐标 |
| 对齐（可选2） | 通过 Procrustes 自动对齐多视角 | `_align_keypoints_to_reference` | 各视角坐标 | 对齐坐标 |
| 聚合 | 对齐后按方法（median/mean/first）融合 | `fuse_3view_keypoints` | (num_views, num_kpts, 3) | (num_kpts, 3) |

---

## 7. 融合策略特点与限制

### 7.1 特点

1. **鲁棒性高**
   - 中位数方法对单视角异常值天然免疫
   - 无需逐个关键点质量评分

2. **灵活性强**
   - 支持有/无相机外参两种模式
   - 支持 Procrustes 无标定对齐

3. **处理缺失视角**
   - 单个关键点可部分视角缺失（只要有 1+ 个视角有效）
   - 但整帧缺失（所有关键点都缺某视角）时需要跳帧

4. **计算快速**
   - 仅对栈进行统计操作
   - O(num_views * num_keypoints) 复杂度

### 7.2 局限

1. **固定三视角**
   - 当前代码围绕 3 个视角设计
   - 扩展到 2 或 4+ 视角需改动

2. **整帧丢弃**
   - 若某视角该帧全无效，整帧跳过
   - 可能损失时序连续性

3. **无动态视角选择**
   - 不支持"每帧自动选最好两个视角"
   - 固定使用所有配置视角

4. **对齐依赖参考视角**
   - Procrustes 对齐质量依赖参考视角质量
   - 若参考视角本身有问题，对齐会失效

---

## 8. 工程建议

1. **监控融合质量**
   - 启用 `enable_fused_view_comparison`
   - 对比融合结果与各单视角差异

2. **缺失视角处理**
   - 记录每帧的 n_valid（有效视角数）
   - 统计缺失比例和位置

3. **参数调试**
   - median 是推荐首选
   - 若噪声大，尝试 procrustes_trimmed 对齐

4. **性能优化**
   - 融合本身 IO 瓶颈比计算更大
   - 重点优化视角数据加载与对齐

---

## 9. 代码快速查找

- 融合主函数：[fuse/fuse.py](fuse/fuse.py) L222-391
- 有效性判断：[fuse/fuse.py](fuse/fuse.py) L39-45
- Procrustes 对齐：[fuse/fuse.py](fuse/fuse.py) L47-107
- 关键点预处理：[infer.py](infer.py) L412-460
- 融合执行入口：[infer.py](infer.py) L182-225

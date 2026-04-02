# 多视角3D关键点融合算法 - LaTeX 数学表述

## 1. 问题定义

给定 $M$ 个视角（本项目中 $M=3$，分别为 front, left, right）对同一场景的 3D 关键点检测结果，每个视角为：

$$\mathbf{K}^{(v)} = \begin{bmatrix}
\mathbf{k}_1^{(v)} \\
\mathbf{k}_2^{(v)} \\
\vdots \\
\mathbf{k}_N^{(v)}
\end{bmatrix} \in \mathbb{R}^{N \times 3}$$

其中：
- $v \in \{1, 2, \ldots, M\}$ 表示视角索引
- $N$ 是关键点总数
- $\mathbf{k}_n^{(v)} = [x_n^{(v)}, y_n^{(v)}, z_n^{(v)}]^\top \in \mathbb{R}^3$ 是第 $v$ 个视角的第 $n$ 个关键点

**融合目标**：得到统一的融合关键点表示 $\mathbf{K}_{\text{fused}} \in \mathbb{R}^{N \times 3}$，使其同时包含多视角的信息。

---

## 2. 有效性判断

### 2.1 单个关键点的有效性

对于第 $v$ 个视角的第 $n$ 个关键点 $\mathbf{k}_n^{(v)}$，定义其有效性指示函数：

$$\text{valid}_{n}^{(v)} = \mathbb{1}_{\text{finite}}(\mathbf{k}_n^{(v)}) \wedge \mathbb{1}_{\text{nonzero}}(\mathbf{k}_n^{(v)})$$

其中：
- $\mathbb{1}_{\text{finite}}(\mathbf{k}_n^{(v)}) = \begin{cases} 1 & \text{if } \mathbf{k}_n^{(v)} \text{ is finite (不含 NaN/Inf)} \\ 0 & \text{otherwise} \end{cases}$

- $\mathbb{1}_{\text{nonzero}}(\mathbf{k}_n^{(v)}) = \begin{cases} 1 & \text{if } \|\mathbf{k}_n^{(v)}\|_2 \geq \epsilon_0 \\ 0 & \text{otherwise} \end{cases}$

- $\epsilon_0$ 是零点阈值（默认 $\epsilon_0 = 10^{-6}$）

### 2.2 融合点的有效性

融合关键点 $\mathbf{k}_n$ 在至少一个视角上有效时，认为该点可融合：

$$\text{fused\_mask}_n = \max_{v \in [1,M]} \text{valid}_{n}^{(v)} = \bigvee_{v=1}^{M} \text{valid}_{n}^{(v)}$$

每个关键点的有效视角数量：

$$\text{n\_valid}_n = \sum_{v=1}^{M} \text{valid}_{n}^{(v)}$$

---

## 3. 坐标系对齐（可选预处理）

若各视角的 3D 关键点已在不同的坐标系中，需先对齐到统一坐标系。

### 3.1 基于已知外参的对齐

如果已知各视角的外部参数（extrinsic parameters），使用变换矩阵 $\mathbf{T}_v = [\mathbf{R}_v | \mathbf{t}_v] \in \mathbb{R}^{3 \times 4}$：

$$\mathbf{k}_n^{(v)}_{\text{world}} = \mathbf{R}_v^\top (\mathbf{k}_n^{(v)}_{\text{camera}} - \mathbf{t}_v)$$

或反向变换（相机到世界）：

$$\mathbf{k}_n^{(v)}_{\text{world}} = \mathbf{R}_v \mathbf{k}_n^{(v)}_{\text{camera}} + \mathbf{t}_v$$

### 3.2 无参数的 Procrustes 对齐

未知外参时，采用 Procrustes 算法自动对齐其他视角到参考视角。

设参考视角为 $v_{\text{ref}}$，对其他视角 $v \neq v_{\text{ref}}$ 进行对齐：

#### 步骤1：确定有效的对应点集

$$\mathcal{P}_{v,\text{ref}} = \{n : \text{valid}_{n}^{(v)} = 1 \wedge \text{valid}_{n}^{(v_{\text{ref}})} = 1\}$$

要求 $|\mathcal{P}_{v,\text{ref}}| \geq 3$（最少需要3个点确定变换）

#### 步骤2：计算均值并中心化

$$\boldsymbol{\mu}_{\text{src}} = \frac{1}{|\mathcal{P}|}  \sum_{n \in \mathcal{P}} \mathbf{k}_n^{(v)}$$
$$\boldsymbol{\mu}_{\text{ref}} = \frac{1}{|\mathcal{P}|}  \sum_{n \in \mathcal{P}} \mathbf{k}_n^{(v_{\text{ref}})}$$

$$\mathbf{K}_{\text{src}}^{\text{c}} = \mathbf{K}_{\text{src}} - \boldsymbol{\mu}_{\text{src}} \in \mathbb{R}^{|\mathcal{P}| \times 3}$$
$$\mathbf{K}_{\text{ref}}^{\text{c}} = \mathbf{K}_{\text{ref}} - \boldsymbol{\mu}_{\text{ref}} \in \mathbb{R}^{|\mathcal{P}| \times 3}$$

#### 步骤3：SVD 求旋转矩阵

计算协方差矩阵的 SVD：

$$\mathbf{H} = (\mathbf{K}_{\text{src}}^{\text{c}})^\top \mathbf{K}_{\text{ref}}^{\text{c}} \in \mathbb{R}^{3 \times 3}$$

$$\mathbf{H} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

旋转矩阵为：

$$\mathbf{R} = \mathbf{V} \mathbf{U}^\top$$

处理反射情况（确保 $\det(\mathbf{R}) = 1$）：

$$\text{if } \det(\mathbf{R}) < 0: \quad \mathbf{V}[-1, :] \leftarrow -\mathbf{V}[-1, :], \quad \mathbf{R} \leftarrow \mathbf{V} \mathbf{U}^\top$$

#### 步骤4：计算缩放因子和平移向量

**缩放因子**（如果允许）：

$$s = \begin{cases}
\frac{\mathrm{tr}(\mathbf{\Sigma})}{|\mathcal{P}|} \cdot \frac{1}{\|(\mathbf{K}_{\text{src}}^{\text{c}})\|_F^2} & \text{if allow\_scale} \\
1 & \text{otherwise}
\end{cases}$$

**平移向量**：

$$\mathbf{t} = \boldsymbol{\mu}_{\text{ref}} - s \cdot \mathbf{R}^\top \boldsymbol{\mu}_{\text{src}}$$

#### 步骤5：应用变换

$$\mathbf{K}^{(v)}_{\text{aligned}} = s \cdot \mathbf{K}^{(v)} \mathbf{R} + \mathbf{t} \mathbb{1}^\top$$

其中 $\mathbb{1} \in \mathbb{R}^N$ 是全1列向量。

### 3.3 Trimmed Procrustes（鲁棒对齐）

处理包含离群点的情况，迭代移除 $p\%$ 的最大误差点：

对于迭代 $i = 1, 2, \ldots, I_{\max}$：

$$\Delta_n = \|s \mathbf{R}^\top \mathbf{k}_n^{(v)} + \mathbf{t} \mathbb{1}^\top - \mathbf{k}_n^{(v_{\text{ref}})}\|_2$$

排序后移除最大的 $\lceil |\mathcal{P}| \cdot p_{\text{trim}} \rceil$ 个点，使用剩余点重新计算变换。

---

## 4. 融合方法

对齐后（或无需对齐时），将 $M$ 个视角的关键点堆叠成张量：

$$\mathcal{K} \in \mathbb{R}^{M \times N \times 3}$$

其中 $\mathcal{K}_{v,n,:} = \mathbf{k}_n^{(v)}$。

### 4.1 中值融合 (Median Fusion)

**仅对有效点进行融合**：

对于每个关键点 $n$，只包含在 $\text{fused\_mask}_n = 1$ 的关键点集合中。将这些点的有效观测值进行中值融合：

$$\mathbf{k}_{n,\text{fused}}^{(d)} = \text{median}\left\{\mathbf{k}_{n}^{(v,d)} : v \in [1,M], \text{valid}_{n}^{(v)} = 1\right\}, \quad d \in \{1,2,3\}$$

其中 $\mathbf{k}_{n}^{(v,d)}$ 表示第 $v$ 个视角、第 $n$ 个关键点的第 $d$ 个坐标。

**矩阵形式**：

设 $\mathcal{K}_{\text{fused}} \in \mathbb{R}^{M \times N \times 3}$ 的第 $n$ 个关键点处，将无效点（$\text{valid}_{n}^{(v)} = 0$）设为 NaN，然后：

$$\mathbf{K}_{\text{fused}}[n] = \text{nanmedian}(\mathcal{K}[:, n, :], \text{axis}=0)$$

这给出 $\mathbf{K}_{\text{fused}}[n] \in \mathbb{R}^3$。

### 4.2 均值融合 (Mean Fusion)

类似地，使用均值代替中值：

$$\mathbf{k}_{n,\text{fused}}^{(d)} = \frac{\sum_{v=1}^{M} \text{valid}_{n}^{(v)} \cdot \mathbf{k}_{n}^{(v,d)}}{\sum_{v=1}^{M} \text{valid}_{n}^{(v)}}$$

矩阵形式：

$$\mathbf{K}_{\text{fused}}[n] = \text{nanmean}(\mathcal{K}[:, n, :], \text{axis}=0)$$

### 4.3 首选融合 (First-Valid Fusion)

取第一个有效视角的观测值：

$$\mathbf{k}_{n,\text{fused}} = \begin{cases}
\mathbf{k}_{n}^{(v^*)}, & \text{where } v^* = \arg\min_v \{v : \text{valid}_{n}^{(v)} = 1\} \\
\text{fill\_value}, & \text{if } \text{valid}_{n}^{(v)} = 0 \text{ for all } v
\end{cases}$$

---

## 5. 综合算法框架

**算法1** 多视角3D关键点融合 (`fuse_3view_keypoints`)

```
输入:
  keypoints_by_view: Dict[str, R^{N×3}]  // 各视角的关键点
  method ∈ {"mean", "median", "first"}  // 融合方法
  alignment_method ∈ {"none", "procrustes", "procrustes_trimmed"}  // 对齐方法
  view_transforms: Optional[Dict]  // 已知外参
  zero_eps: float = 1e-6  // 零点阈值
  
输出:
  K_fused ∈ R^{N×3}  // 融合后的关键点
  fused_mask ∈ {0,1}^N  // 有效性掩码
  n_valid ∈ Z^N  // 每个点的有效视角数
```

**步骤1** 坐标系对齐

```
if view_transforms is not None:
  // 使用已知外参对齐
  for v in view_list:
    K^{(v)} ← T_v · K^{(v)}  
elif alignment_method in {"procrustes", "procrustes_trimmed"}:
  // 使用 Procrustes 对齐
  v_ref ← 参考视角 (默认为第1个)
  for v in view_list \ {v_ref}:
    if alignment_method == "procrustes":
      K^{(v)} ← Procrustes_align(K^{(v_ref)}, K^{(v)})
    else:
      K^{(v)} ← Procrustes_trimmed_align(K^{(v_ref)}, K^{(v)}, p_trim)
```

**步骤2** 堆叠和有效性判断

```
K_stack ∈ R^{M×N×3}
for v ∈ [1, M]:
  K_stack[v] ← K^{(v)}

for n ∈ [1, N]:
  for v ∈ [1, M]:
    valid[v, n] ← is_finite(K_stack[v, n]) ∧ (||K_stack[v, n]||_2 ≥ ε_0)
  
  fused_mask[n] ← max_v{valid[v, n]}
  n_valid[n] ← Σ_v valid[v, n]
```

**步骤3** 融合

```
K_fused ← full((N, 3), fill_value)

// 仅融合有效点
valid_indices ← {n : fused_mask[n] = 1}

if method == "median":
  for n in valid_indices:
    for d in {1, 2, 3}:
      K_fused[n, d] ← median([K_stack[v, n, d] : v where valid[v, n]])
      
elif method == "mean":
  for n in valid_indices:
    for d in {1, 2, 3}:
      K_fused[n, d] ← mean([K_stack[v, n, d] : v where valid[v, n]])
      
elif method == "first":
  for n in valid_indices:
    for v in [1, M]:
      if valid[v, n]:
        K_fused[n] ← K_stack[v, n]
        break
```

---

## 6. 数学属性分析

### 6.1 中值融合的鲁棒性

**定理**：若在 $M$ 个视角中至少有 $\lceil M/2 \rceil$ 个观测有效，中值融合对其余观测中的异常值不敏感。

**证明思路**：
- 离群点最多占 $\lfloor M/2 \rfloor$ 个
- 中值选择至少 $\lceil M/2 \rceil$ 个最小观测值的中位数
- 因此对 $\lfloor M/2 \rfloor$ 个离群点具有鲁棒性

### 6.2 均值融合的偏差分析

假设各视角的观测误差相互独立且满足 $\mathbf{e}^{(v)} \sim \mathcal{N}(0, \sigma_v^2 \mathbf{I})$，则融合后的协方差为：

$$\mathbf{\Sigma}_{\text{fused}} = \left(\sum_{v=1}^{M} \sigma_v^{-2}\right)^{-1} \mathbf{I}$$

若各视角精度相等 ($\sigma_v = \sigma$)，则：

$$\mathbf{\Sigma}_{\text{fused}} = \frac{\sigma^2}{M} \mathbf{I}$$

即融合后的标准差减少为 $1/\sqrt{M}$。

### 6.3 Procrustes 对齐的最优性

**定理（正交Procrustes问题）**：
给定两个点集 $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{N \times 3}$，求旋转矩阵 $\mathbf{R}$ 和平移 $\mathbf{t}$ 最小化：

$$\min_{\mathbf{R}, \mathbf{t}} \|\mathbf{A}(\mathbf{R}^\top) + \mathbf{t}\mathbb{1}^\top - \mathbf{B}\|_F^2$$

其中 $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$ 且 $\det(\mathbf{R}) = 1$。

**解**：通过 SVD $(\mathbf{A}^c)^\top \mathbf{B}^c = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ 得到 $\mathbf{R}^* = \mathbf{V}\mathbf{U}^\top$（以及平移 $\mathbf{t}^* = \bar{\mathbf{b}} - (\bar{\mathbf{a}})\mathbf{R}^*$）。

---

## 7. 计算复杂度分析

| 步骤 | 操作 | 复杂度 |
|------|------|--------|
| 有效性判断 | 检查 $M \times N$ 个点 | $O(MN)$ |
| Procrustes 对齐 | SVD ($|\mathcal{P}| \leq N$) | $O(N^2)$ per view |
| 融合（均值/中值） | 对 $N$ 个点求平均/中值 | $O(MN \log N)$ (中值)<br> $O(MN)$ (均值) |
| **总计** | | $O(M(N^2 + N \log N))$ |

对于典型参数 ($M=3$, $N \approx 70$)：
- 无对齐：~$250$ 次浮点运算
- 含 Procrustes：~$15,000$ 次浮点运算
- 实时性：可在 ms 级完成

---

## 8. 实现注意事项

### 8.1 数值稳定性

1. **避免直接零值比较**：使用 $\|\mathbf{k}\|_2 \geq \epsilon_0$ 而非 $\|\mathbf{k}\|_2 > 0$
   
2. **NaN 处理**：融合前将无效点置为 NaN，使用 `nanmean`/`nanmedian`

3. **SVD 中的条件数**：若 $\kappa(\mathbf{H}) \gg 1$，考虑增加对齐点数或使用预处理

### 8.2 边界情况

| 情况 | 处理方式 |
|------|---------|
| 单个视角有效 | 直接返回该视角的观测 |
| 所有视角无效 | 填充 NaN 或指定默认值 |
| 对齐点数不足 | 跳过对齐，发出警告 |
| 包含反射的旋转矩阵 | 调整 $\mathbf{V}$ 的最后一行 |

---

## LaTeX 公式速查表

### 核心融合公式

**中值融合**：
```latex
\mathbf{k}_{n,\mathrm{fused}}^{(d)} = \mathrm{median}\left\{\mathbf{k}_{n}^{(v,d)} : v \in [1,M], \mathrm{valid}_{n}^{(v)} = 1\right\}
```

**均值融合**：
```latex
\mathbf{k}_{n,\mathrm{fused}}^{(d)} = \frac{\sum_{v=1}^{M} \mathrm{valid}_{n}^{(v)} \cdot \mathbf{k}_{n}^{(v,d)}}{\sum_{v=1}^{M} \mathrm{valid}_{n}^{(v)}}
```

**有效性判断**：
```latex
\mathrm{valid}_{n}^{(v)} = \mathbb{1}_{\mathrm{finite}}(\mathbf{k}_n^{(v)}) \wedge \mathbb{1}_{\mathrm{nonzero}}(\mathbf{k}_n^{(v)})
```

**Procrustes 旋转**：
```latex
\mathbf{R}^{*} = \mathbf{V}\mathbf{U}^{\top} \quad \text{where} \quad \mathbf{H} = (\mathbf{K}_{\mathrm{src}}^{\mathrm{c}})^{\top}\mathbf{K}_{\mathrm{ref}}^{\mathrm{c}} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^{\top}
```

---

## 完整示例

给定三个视角的关键点：

$$\mathbf{K}^{(\text{front})} = \begin{bmatrix} 0.1 & 0.2 & 0.5 \\ 0.2 & 0.3 & 0.6 \\ \vdots \end{bmatrix}, \quad \mathbf{K}^{(\text{left})} = \begin{bmatrix} 0.15 & 0.25 & 0.52 \\ \text{NaN} & \text{NaN} & \text{NaN} \\ \vdots \end{bmatrix}$$

对第1个关键点进行融合（中值）：
1. 有效观测：front: $[0.1, 0.2, 0.5]$, left: $[0.15, 0.25, 0.52]$
2. 去除 NaN 后计算中值：
   - $x$: $\text{median}(0.1, 0.15) = 0.125$
   - $y$: $\text{median}(0.2, 0.25) = 0.225$
   - $z$: $\text{median}(0.5, 0.52) = 0.51$
3. 融合结果：$\mathbf{k}_{1,\text{fused}} = [0.125, 0.225, 0.51]^\top$

对第2个关键点（left 无效）：
1. 仅 front 有效：$\mathbf{k}_{2,\text{fused}} = \mathbf{k}_2^{(\text{front})}$


# Multi-View 3D Keypoint Fusion Algorithm
## A Formal Mathematical Treatment for Publication

---

## 1. Problem Formulation

Given $M$ views (in this work, $M=3$: front, left, right) observing the same scene, each view provides a set of detected 3D keypoints:

$$\mathbf{K}^{(v)} = \begin{bmatrix}
\mathbf{k}_1^{(v)} \\
\mathbf{k}_2^{(v)} \\
\vdots \\
\mathbf{k}_N^{(v)}
\end{bmatrix} \in \mathbb{R}^{N \times 3}$$

where:
- $v \in \{1, 2, \ldots, M\}$ denotes the view index
- $N$ is the total number of keypoints
- $\mathbf{k}_n^{(v)} = [x_n^{(v)}, y_n^{(v)}, z_n^{(v)}]^\top \in \mathbb{R}^3$ represents the $n$-th keypoint in the $v$-th view

**Objective:** Obtain a unified fused keypoint representation $\mathbf{K}_{\text{fused}} \in \mathbb{R}^{N \times 3}$ that incorporates multi-view information coherently.

---

## 2. Validity Assessment

### 2.1 Point-Level Validity

For a keypoint $\mathbf{k}_n^{(v)}$ in view $v$, we define its validity indicator as:

$$\text{valid}_{n}^{(v)} = \mathbb{1}_{\text{finite}}(\mathbf{k}_n^{(v)}) \land \mathbb{1}_{\text{nonzero}}(\mathbf{k}_n^{(v)})$$

where:

$$\mathbb{1}_{\text{finite}}(\mathbf{k}_n^{(v)}) = \begin{cases} 1 & \text{if all components are finite (no NaN/Inf)} \\ 0 & \text{otherwise} \end{cases}$$

$$\mathbb{1}_{\text{nonzero}}(\mathbf{k}_n^{(v)}) = \begin{cases} 1 & \text{if } \|\mathbf{k}_n^{(v)}\|_2 \geq \epsilon_0 \\ 0 & \text{otherwise} \end{cases}$$

Here $\epsilon_0$ is a negligibility threshold (default: $\epsilon_0 = 10^{-6}$) that filters out near-zero detections, which typically indicate tracking failures.

### 2.2 Fusion-Level Validity

A fused keypoint is considered valid if it is observed validly in at least one view:

$$\text{fused\_mask}_n = \bigvee_{v=1}^{M} \text{valid}_{n}^{(v)}$$

For each keypoint, we also track the number of valid observations:

$$\text{n\_valid}_n = \sum_{v=1}^{M} \text{valid}_{n}^{(v)}$$

---

## 3. Coordinate System Alignment (Optional Preprocessing)

If the 3D keypoints from different views are expressed in different coordinate systems, alignment to a common frame is necessary before fusion.

### 3.1 Transformation Using Known Extrinsics

When camera extrinsic parameters are available, we apply the transformation:

$$\mathbf{k}_n^{(v)}_{\text{world}} = \mathbf{R}_v \mathbf{k}_n^{(v)}_{\text{camera}} + \mathbf{t}_v$$

where $\mathbf{R}_v \in SO(3)$ and $\mathbf{t}_v \in \mathbb{R}^3$ are the rotation and translation of view $v$ relative to the world frame.

### 3.2 Procrustes-Based Alignment (Unknown Extrinsics)

When extrinsics are unavailable, we employ the Procrustes algorithm to automatically align non-reference views to a reference view $v_{\text{ref}}$.

#### Step 1: Identify Valid Correspondence Set

$$\mathcal{P}_{v \to v_{\text{ref}}} = \{n : \text{valid}_{n}^{(v)} = 1 \land \text{valid}_{n}^{(v_{\text{ref}})} = 1\}$$

Require $|\mathcal{P}| \geq 3$ for a valid pose estimate.

#### Step 2: Compute Centroids and Center Points

$$\boldsymbol{\mu}_{\text{src}} = \frac{1}{|\mathcal{P}|} \sum_{n \in \mathcal{P}} \mathbf{k}_n^{(v)}$$

$$\boldsymbol{\mu}_{\text{ref}} = \frac{1}{|\mathcal{P}|} \sum_{n \in \mathcal{P}} \mathbf{k}_n^{(v_{\text{ref}})}$$

$$\mathbf{K}_{\text{src}}^{c} = \{\mathbf{k}_n^{(v)} - \boldsymbol{\mu}_{\text{src}} : n \in \mathcal{P}\} \in \mathbb{R}^{|\mathcal{P}| \times 3}$$

$$\mathbf{K}_{\text{ref}}^{c} = \{\mathbf{k}_n^{(v_{\text{ref}})} - \boldsymbol{\mu}_{\text{ref}} : n \in \mathcal{P}\} \in \mathbb{R}^{|\mathcal{P}| \times 3}$$

#### Step 3: Compute Optimal Rotation via SVD

Perform the singular value decomposition of the Gram matrix:

$$\mathbf{H} = (\mathbf{K}_{\text{src}}^{c})^\top \mathbf{K}_{\text{ref}}^{c} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

The optimal rotation matrix is:

$$\mathbf{R}^* = \mathbf{V} \mathbf{U}^\top$$

To ensure a proper rotation (not a reflection), we check and correct:

$$\text{if } \det(\mathbf{R}^*) < 0: \quad \mathbf{V}[-1, :] \gets -\mathbf{V}[-1, :], \quad \mathbf{R}^* \gets \mathbf{V} \mathbf{U}^\top$$

#### Step 4: Estimate Scale (if applicable) and Translation

**Scale factor** (only when scale estimation is enabled):

$$s^* = \begin{cases}
\frac{\sum_{i} \sigma_i}{\|\mathbf{K}_{\text{src}}^{c}\|_F^2} & \text{if } \text{allow\_scale} = \text{True} \\
1.0 & \text{otherwise}
\end{cases}$$

where $\sigma_i$ are the singular values and $\|\cdot\|_F$ denotes the Frobenius norm.

**Translation vector**:

$$\mathbf{t}^* = \boldsymbol{\mu}_{\text{ref}} - s^* (\mathbf{R}^*)^\top \boldsymbol{\mu}_{\text{src}}$$

#### Step 5: Apply Transformation

$$\mathbf{K}^{(v)}_{\text{aligned}} = s^* \mathbf{K}^{(v)} (\mathbf{R}^*)^\top + \mathbf{t}^* \mathbb{1}^\top$$

where $\mathbb{1} \in \mathbb{R}^N$ is the all-ones vector.

### 3.3 Trimmed Procrustes (Robust Alignment with Outliers)

For robustness against outliers, we iteratively remove the $p\%$ of points with largest alignment errors:

For iteration $i = 1, 2, \ldots, I_{\max}$:

$$\Delta_n = \|s^* (\mathbf{R}^*)^\top \mathbf{k}_n^{(v)} + \mathbf{t}^* - \mathbf{k}_n^{(v_{\text{ref}})}\|_2$$

Remove the $\lceil |\mathcal{P}| \cdot p_{\text{trim}} \rceil$ points with largest $\Delta_n$, and recompute $(s^*, \mathbf{R}^*, \mathbf{t}^*)$ using the remaining inliers.

---

## 4. Fusion Strategies

After alignment (or without alignment if not applicable), we stack the keypoints from all views into a tensor:

$$\mathcal{K} \in \mathbb{R}^{M \times N \times 3}, \quad \mathcal{K}_{v,n,:} = \mathbf{k}_n^{(v)}_{\text{aligned}}$$

### 4.1 Median Fusion

For each keypoint $n$, compute the coordinate-wise median over all valid observations:

$$\mathbf{k}_{n,\text{fused}}^{(d)} = \text{median}\left(\left\{\mathbf{k}_{n}^{(v,d)} : v \in [1,M], \text{valid}_{n}^{(v)} = 1\right\}\right), \quad d \in \{1,2,3\}$$

**Matrix-vector form:**

$$\mathbf{K}_{\text{fused}}[n] = \text{nanmedian}(\mathcal{K}[:, n, :], \text{axis}=0)$$

where invalid observations are represented as NaN and excluded from the median computation.

### 4.2 Mean Fusion

Compute the weighted arithmetic mean over valid observations:

$$\mathbf{k}_{n,\text{fused}}^{(d)} = \frac{1}{\text{n\_valid}_n} \sum_{v=1}^{M} \text{valid}_{n}^{(v)} \cdot \mathbf{k}_{n}^{(v,d)}$$

**Matrix-vector form:**

$$\mathbf{K}_{\text{fused}}[n] = \text{nanmean}(\mathcal{K}[:, n, :], \text{axis}=0)$$

### 4.3 First-Valid Fusion

Select the observation from the first valid view:

$$\mathbf{k}_{n,\text{fused}} = \begin{cases}
\mathbf{k}_{n}^{(v^*)}, & \text{where } v^* = \min\{v : \text{valid}_{n}^{(v)} = 1\} \\
\text{fill\_value}, & \text{if } \text{n\_valid}_n = 0
\end{cases}$$

This strategy preserves the original detection without any averaging.

---

## 5. Formal Algorithm Specification

**Algorithm 1: Multi-View 3D Keypoint Fusion**

```
Input:
  K_by_view: Dict[str → R^{N×3}]              // Keypoints from each view
  method ∈ {"mean", "median", "first"}        // Fusion strategy
  align_method ∈ {"none", "procrustes", "procrustes_trimmed"}
  view_transforms: Optional[Dict]             // Known extrinsic parameters
  ε₀: float = 1e-6                            // Negligibility threshold
  fill_value: float = NaN                     // Filler for invalid keypoints

Output:
  K_fused ∈ R^{N×3}                           // Fused keypoints
  fused_mask ∈ {0,1}^N                        // Validity mask
  n_valid ∈ Z_≥0^N                            // Count of valid observations
```

**Procedure:**

**1. Coordinate Alignment**

```
if view_transforms ≠ null:
  // Apply known extrinsics
  for each view v:
    K^{(v)} ← apply_transformation(K^{(v)}, T_v)
elif align_method ∈ {"procrustes", "procrustes_trimmed"}:
  // Automatic Procrustes alignment
  v_ref ← reference_view (default: 1st view)
  for each view v ≠ v_ref:
    K^{(v)} ← procrustes_align(K^{(v_ref)}, K^{(v)})
```

**2. Validity Check and Stacking**

```
K_stack ∈ R^{M×N×3}
for v ∈ {1, ..., M}:
  for n ∈ {1, ..., N}:
    valid[v,n] ← is_finite(K_stack[v,n]) ∧ (‖K_stack[v,n]‖₂ ≥ ε₀)

for n ∈ {1, ..., N}:
  fused_mask[n] ← max_v{valid[v,n]}
  n_valid[n] ← Σ_v valid[v,n]
```

**3. Fusion**

```
K_fused ∈ R^{N×3}, initialize as fill_value

for n where fused_mask[n] = 1:
  if method == "median":
    K_fused[n] ← nanmedian({K_stack[v,n] : valid[v,n]=1})
  elif method == "mean":
    K_fused[n] ← nanmean({K_stack[v,n] : valid[v,n]=1})
  elif method == "first":
    K_fused[n] ← K_stack[v*,n] where v* = argmin{v : valid[v,n]=1}

return (K_fused, fused_mask, n_valid)
```

---

## 6. Mathematical Properties

### 6.1 Robustness of Median Fusion

**Proposition 1 (Breakdown Point):** If at least $\lceil M/2 \rceil$ observations are valid, median fusion is robust to outliers in the remaining $\lfloor M/2 \rfloor$ observations.

*Proof sketch:* The median selects the middle value; thus, at most $\lfloor M/2 \rfloor$ arbitrary corruptions cannot change the median when $\lceil M/2 \rceil$ inliers exist. □

### 6.2 Variance Reduction in Mean Fusion

**Proposition 2 (Noise Reduction):** Assuming observations are corrupted by independent, isotropic Gaussian noise $\mathbf{e}^{(v)} \sim \mathcal{N}(\mathbf{0}, \sigma_v^2 \mathbf{I})$, the fused estimate inherits covariance:

$$\mathbf{\Sigma}_{\text{fused}} = \left(\sum_{v=1}^{M} \sigma_v^{-2}\right)^{-1} \mathbf{I}$$

In the homoscedastic case ($\sigma_v = \sigma$ for all $v$):

$$\mathbf{\Sigma}_{\text{fused}} = \frac{\sigma^2}{M} \mathbf{I}$$

i.e., the standard deviation decreases by a factor of $1/\sqrt{M}$. □

### 6.3 Optimality of Procrustes Alignment

**Theorem 1 (Orthogonal Procrustes Problem):** Given two point sets $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{N \times 3}$, the rotation matrix $\mathbf{R}^* \in SO(3)$ and translation $\mathbf{t}^*$ that minimize

$$J(\mathbf{R}, \mathbf{t}) = \left\|\mathbf{A}(\mathbf{R}^\top) + \mathbf{t}\mathbb{1}^\top - \mathbf{B}\right\|_F^2$$

are uniquely given by:
1. $\mathbf{R}^* = \mathbf{V}\mathbf{U}^\top$ where $(\mathbf{A}^c)^\top\mathbf{B}^c = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ (SVD)
2. Adjust if $\det(\mathbf{R}^*) = -1$: set $\mathbf{V}[-1,:] \gets -\mathbf{V}[-1,:]$ and recompute
3. $\mathbf{t}^* = \bar{\mathbf{b}} - (\bar{\mathbf{a}})(\mathbf{R}^*)^\top$

*Reference:* Kabsch, W. (1976). Acta Crystallographica.

---

## 7. Computational Complexity

| Operation | Time Complexity |
|-----------|-----------------|
| Validity assessment | $O(MN)$ |
| Procrustes SVD per view | $O(N^2)$ |
| Coordinate alignment | $O(MN^2)$ |
| Fusion (median/mean) | $O(MN \log M)$ / $O(MN)$ |
| **Total** | $O(MN^2)$ or $O(MN \log M)$ |

For typical parameters ($M=3$, $N \approx 70$):
- Without alignment: $\sim 250$ FLOPs
- With Procrustes: $\sim 15,000$ FLOPs
- **Execution time:** < 1 ms per frame on modern CPU

---

## 8. Experimental Validation

### 8.1 Setup

- **Dataset:** Multi-view driver action sequences (3 synchronized RGB cameras)
- **Baseline Methods Compared:**
  1. Single-view: using front view only
  2. Early Fusion: averaging keypoints without alignment
  3. Mean Fusion (our method, unaligned)
  4. Median Fusion (our method, unaligned)
  5. Median Fusion + Procrustes (our method, aligned)

### 8.2 Evaluation Metrics

**Metric 1: Consensus Consistency**
$$\text{CC}_n = \frac{1}{M(M-1)/2} \sum_{1 \leq v < u \leq M} \|\mathbf{k}_n^{(v)} - \mathbf{k}_n^{(u)}\|_2$$

Lower values indicate higher agreement among views.

**Metric 2: Temporal Smoothness (before further temporal filtering)**
$$\text{TS} = \frac{1}{T-2} \sum_{t=2}^{T-1} \left\|\frac{\partial^2 \mathbf{K}_t}{\partial t^2}\right\|_F$$

**Metric 3: Stability Against View Dropout**
Evaluate robustness when one view is removed; acceptable degradation should be $< 10\%$.

---

## 9. Implementation Details and Numerical Stability

### 9.1 Handling NaN and Infinity

1. **Invalid Detection Detection:** Points with any coordinate being NaN/Inf are marked invalid.
2. **Zero-Clipping:** Points with $\|\mathbf{k}\|_2 < \epsilon_0$ are clipped to fill_value (NaN).
3. **Fusion:** Use `np.nanmean()` and `np.nanmedian()` to automatically exclude NaN entries.

### 9.2 SVD Numerical Stability

When computing $\mathbf{H} = (\mathbf{K}_{\text{src}}^{c})^\top\mathbf{K}_{\text{ref}}^{c}$:
- Ensure centered points: $\mathbf{K}^c = \mathbf{K} - \bar{\mathbf{K}}$
- Monitor condition number $\kappa(\mathbf{H})$; if $\kappa > 10^{10}$, consider regularization or augmenting correspondence set

### 9.3 Handling Edge Cases

| Case | Handling |
|------|----------|
| Insufficient valid points ($|\mathcal{P}| < 3$) | Skip alignment; issue warning |
| No valid observations in all views | Return NaN for that keypoint |
| All observations identical | Scale factor $s^* = 1$ (well-defined) |
| Reflection in SVD | Flip appropriate column of $\mathbf{V}$ |

---

## 10. Recommended Configuration

```yaml
# Typical settings for automotive driver action dataset
fusion:
  method: "median"                              # Robust to outliers
  alignment_method: "procrustes"                # Automatic alignment
  alignment_reference: "front"                  # Front view as reference
  alignment_scale: false                        # Rigid body assumption
  alignment_trim_ratio: 0.20                    # For trimmed variant
  zero_epsilon: 1.0e-6                          # Negligibility threshold
  fill_value: NaN                               # Missing value indicator
```

**Rationale:**
- **Median:** Most robust to occasional misdetections
- **Procrustes:** Handles camera misalignment without calibration
- **No scale:** Assumes rigid body (suitable for human head region)
- **Trim ratio:** 20% outlier removal balances robustness and inlier usage

---

## 11. Code Snippet for Reproduction

```python
import numpy as np

def fuse_3view_keypoints(keypoints_by_view, method="median", 
                         alignment_method="procrustes"):
    """
    Fuse 3D keypoints from multiple views.
    
    Args:
        keypoints_by_view: Dict[str, np.ndarray] of shape (N, 3)
        method: "mean", "median", or "first"
        alignment_method: "none", "procrustes", "procrustes_trimmed"
    
    Returns:
        K_fused, fused_mask, n_valid
    """
    M = len(keypoints_by_view)
    views = sorted(keypoints_by_view.keys())
    
    # Step 1: Alignment (optional)
    if alignment_method in ["procrustes", "procrustes_trimmed"]:
        ref_view = views[0]
        for v in views[1:]:
            keypoints_by_view[v] = procrustes_align(
                keypoints_by_view[ref_view],
                keypoints_by_view[v]
            )
    
    # Step 2: Stack and compute validity
    K_stack = np.stack([keypoints_by_view[v] for v in views], axis=0)
    
    finite = np.isfinite(K_stack).all(axis=-1)  # (M, N)
    nonzero = np.linalg.norm(K_stack, axis=-1) >= 1e-6  # (M, N)
    valid = finite & nonzero
    
    fused_mask = valid.any(axis=0)
    n_valid = valid.sum(axis=0)
    
    # Step 3: Fusion
    K_fused = np.full((K_stack.shape[1], 3), np.nan)
    
    K_masked = K_stack.copy()
    K_masked[~valid] = np.nan
    
    if method == "median":
        K_fused[fused_mask] = np.nanmedian(K_stack[:, fused_mask, :], axis=0)
    elif method == "mean":
        K_fused[fused_mask] = np.nanmean(K_stack[:, fused_mask, :], axis=0)
    elif method == "first":
        for n in range(K_stack.shape[1]):
            for v in range(M):
                if valid[v, n]:
                    K_fused[n] = K_stack[v, n]
                    break
    
    return K_fused, fused_mask.astype(int), n_valid
```

---

## 12. Journal Publication Guidance

### For Methods Section:

Use sections 2–5 as-is. Adjust terminology to match your paper's conventions.

### For Results:

Report median/mean errors for each fusion method against:
- Ground truth (if available from motion capture)
- Consensus among views (Metric 1, Section 8.2)
- Temporal smoothness before and after temporal filtering (Metric 2)

### For Discussion:

- Explain why median was superior (robustness proposition, Section 6.1)
- Discuss computational efficiency (Section 7)
- Address failure modes and mitigation strategies (Section 9.3)

---

## References (Template)

Kabsch, W. (1976). A solution for the best rotation to relate two sets of vectors. *Acta Crystallographica Section A*, 32(5), 922–923.

Gower, J. C. (1975). Generalized Procrustes analysis. *Psychometrika*, 40(1), 33–51.

Chen, K., et al. (2026). Multi-view 3D keypoint fusion for driver head pose estimation. [Your Venue].

---

## Summary Table: Quick Reference

| **Aspect** | **Details** |
|-----------|-----------|
| **Input** | $M$ views of $N$ 3D keypoints each |
| **Validity Check** | Finite \& $\|\mathbf{k}\|_2 \geq \epsilon_0$ |
| **Alignment Options** | Known extrinsics, Procrustes, Procrustes-trimmed |
| **Fusion Methods** | Median (robust), Mean (optimal under Gaussian noise), First (preserves original) |
| **Complexity** | $O(MN^2)$ with Procrustes; $O(MN)$ median fusion alone |
| **Robustness** | Median robust to $\lfloor M/2 \rfloor$ corruptions |
| **Output** | Fused keypoints, validity mask, observation count per point |


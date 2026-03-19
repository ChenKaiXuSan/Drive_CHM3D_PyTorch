import numpy as np
from scipy import interpolate
import logging
logger = logging.getLogger(__name__)

# ---------- 工具函数 ----------

def _longest_run(mask_1d: np.ndarray) -> int:
    """返回布尔序列中最长连续 True 的长度。"""
    if mask_1d.size == 0:
        return 0
    x = np.concatenate(([0], mask_1d.view(np.int8), [0]))
    diff = np.diff(x)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    return int((ends - starts).max()) if starts.size else 0

def _interp_linear_no_extrap(t_valid, y_valid, t_all):
    """线性插值；首尾不外推，改为贴边（edge-hold）。"""
    y = np.interp(t_all, t_valid, y_valid)
    y[t_all < t_valid[0]] = y_valid[0]
    y[t_all > t_valid[-1]] = y_valid[-1]
    return y

def _interp_pchip_no_extrap(t_valid, y_valid, t_all):
    """PCHIP 插值；首尾贴边。"""
    f = interpolate.PchipInterpolator(t_valid, y_valid, extrapolate=False)
    y = f(t_all)
    # PCHIP 对外侧返回 nan；改为贴边
    if np.isnan(y[0]):
        y[:np.argmax(t_all >= t_valid[0])] = y_valid[0]
    if np.isnan(y[-1]):
        y[np.argmax(t_all > t_valid[-1]):] = y_valid[-1]
    return np.asarray(y, dtype=float)

def _ffill_bfill(y: np.ndarray) -> np.ndarray:
    """先前向再后向填充；全 NaN 则仍为 NaN。"""
    y = y.copy()
    m = np.isfinite(y)
    if not m.any():
        return y
    # ffill
    idx = np.where(m, np.arange(len(y)), 0)
    np.maximum.accumulate(idx, out=idx)
    y_ff = y[idx]
    # bfill
    idx2 = np.where(m, np.arange(len(y)), len(y)-1)
    idx2 = np.minimum.accumulate(idx2[::-1])[::-1]
    y_bf = y[idx2]
    # 合并：优先 ffill，缺的用 bfill
    out = y_ff
    out[~m] = y_bf[~m]
    return out

# ---------- 缺失检测 ----------

def detect_missing_frames(X_seq: np.ndarray,
                          zero_as_missing: bool = True,
                          zero_axes: tuple = (0, 1, 2),
                          atol: float = 1e-6):
    """
    检测逐帧是否缺失（全 NaN 或(可选)近零）。
    zero_axes: 指定哪些坐标轴为“近零即缺失”的判定（例如仅 (0,1) 表示 xy 近零算缺失，z 不算）
    """
    # 任一坐标为 NaN → 该帧视为缺
    is_nan = np.isnan(X_seq).any(axis=(1, 2))
    if zero_as_missing:
        Xz = X_seq[:, :, list(zero_axes)]
        is_zero = np.all(np.isfinite(Xz), axis=(1, 2)) & np.allclose(Xz, 0, atol=atol)
        missing = np.logical_or(is_nan, is_zero)
    else:
        missing = is_nan
    return missing

# ---------- 主插值函数（改进版） ----------

def interpolate_missing_frames(
    X_seq: np.ndarray,
    method: str = "auto",
    zero_as_missing: bool = True,
    zero_axes: tuple = (0, 1, 2),
    s_spline: float = 1e-6,
):
    """
    对缺失帧进行插值修复。
    method: "linear" | "pchip" | "spline" | "ffill" | "ffill_bfill" | "auto"
    zero_as_missing: True 时将近零值视为缺失（受 zero_axes 限制）
    返回:
      X_interp: (T,J,3)
      info: dict，包含 missing_mask / valid_counts / longest_missing_run 等
    """
    X = np.array(X_seq, copy=True, dtype=float)
    T, J, C = X.shape
    t_all = np.arange(T, dtype=float)

    missing = detect_missing_frames(X, zero_as_missing=zero_as_missing, zero_axes=zero_axes)
    miss_ratio = float(np.mean(missing))
    # 连续缺失最长段（逐关节取最大）
    longest_run = 0
    for j in range(J):
        # 该关节“整帧缺失”定义：该帧此关节的任一坐标为 nan 或(可选)近零
        mj = np.isnan(X[:, j, :]).any(axis=1)
        longest_run = max(longest_run, _longest_run(mj))

    # auto 策略：先看总体缺失率，再看最长连续缺失
    if method == "auto":
        if miss_ratio <= 0.05 and longest_run <= 3:
            method_eff = "pchip"   # 少量缺 + 短缺段，用 PCHIP，平滑且不过冲
        elif miss_ratio <= 0.15 and longest_run <= 6:
            method_eff = "linear"  # 中等缺失，用线性更稳，不会振铃
        elif miss_ratio <= 0.35:
            method_eff = "ffill_bfill"  # 缺失较多，优先稳妥
        else:
            method_eff = "ffill"        # 极端情况下保守
    else:
        method_eff = method.lower()

    # 逐关节逐通道插值
    valid_counts = np.zeros((J, C), dtype=int)
    for j in range(J):
        for c in range(C):
            y = X[:, j, c]
            # 有效点：非 nan 且（可选）非近零
            if zero_as_missing and (c in zero_axes):
                valid = np.isfinite(y) & (~np.isclose(y, 0.0))
            else:
                valid = np.isfinite(y)
            n_valid = int(valid.sum())
            valid_counts[j, c] = n_valid

            if n_valid == 0:
                # 全缺：保持 NaN
                continue

            t_valid = t_all[valid]
            y_valid = y[valid]

            # 去重（罕见情况下时间戳重复）
            if t_valid.size > 1:
                tu, inv = np.unique(t_valid, return_inverse=True)
                if tu.size != t_valid.size:
                    buf = np.zeros(tu.size, dtype=float)
                    cnt = np.zeros(tu.size, dtype=int)
                    for k, val in enumerate(y_valid):
                        buf[inv[k]] += val
                        cnt[inv[k]] += 1
                    y_valid = buf / np.maximum(cnt, 1)
                    t_valid = tu

            # 实际插值
            if method_eff == "ffill":
                y_interp = _ffill_bfill(y) if n_valid < T else y  # 全有则不变
            elif method_eff == "ffill_bfill":
                y_interp = _ffill_bfill(y)
            elif method_eff == "pchip" and t_valid.size >= 2:
                y_interp = _interp_pchip_no_extrap(t_valid, y_valid, t_all)
            elif method_eff == "spline" and t_valid.size >= 4:
                # 注意：样条可能过冲；作为备选
                k = int(min(3, t_valid.size - 1))
                try:
                    f = interpolate.UnivariateSpline(t_valid, y_valid, k=k, s=s_spline)
                    y_interp = f(t_all)
                    # 首尾贴边
                    y_interp[:np.argmax(t_all >= t_valid[0])] = y_valid[0]
                    y_interp[np.argmax(t_all > t_valid[-1]):] = y_valid[-1]
                except Exception:
                    y_interp = _interp_linear_no_extrap(t_valid, y_valid, t_all)
            else:
                # 默认线性 + 贴边
                y_interp = _interp_linear_no_extrap(t_valid, y_valid, t_all)

            # 写回（仅覆盖缺失/近零处；保留原本有效观测）
            if zero_as_missing and (c in zero_axes):
                write_mask = ~np.isfinite(y) | np.isclose(y, 0.0)
            else:
                write_mask = ~np.isfinite(y)
            y[write_mask] = y_interp[write_mask]
            X[:, j, c] = y

    info = {
        "missing_mask": missing,              # (T,)
        "miss_ratio": float(miss_ratio),
        "longest_missing_run": int(longest_run),
        "valid_counts": valid_counts,         # (J,3)
        "method_effective": method_eff,
    }
    logger.info(f"[interp] miss={miss_ratio:.3f}, longest_run={longest_run}, mode={method_eff}")
    return X, info
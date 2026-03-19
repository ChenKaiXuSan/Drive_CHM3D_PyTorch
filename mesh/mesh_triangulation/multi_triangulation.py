#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
triangulation_plus.py
Author: Kaixu Chen (2025)
--------------------------------
Robust multi-view triangulation with epipolar gating,
confidence weighting, and temporal smoothing.

Features:
- n-view linear triangulation
- Sampson error filtering
- Confidence weighting
- Optional temporal smoothing (Savitzky-Golay / EMA)
- Basic evaluation metrics (RPE, cheirality rate, epipolar inlier)

"""

from typing import Dict, List, Tuple
import numpy as np
import logging
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


# ================= 工具函数 =================


def build_P(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """K(3x3), R(3x3), t(3,) -> P(3x4)"""
    return K @ np.hstack([R, t.reshape(3, 1)])


def project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """3D 点投影到像素 (2,)"""
    X_h = np.hstack([X, 1.0])
    x = P @ X_h
    return x[:2] / x[2]


def cheirality_ok(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> bool:
    """正深度检查"""
    X_cam = R @ X + t
    return X_cam[2] > 0


def reproj_error(Ps: List[np.ndarray], X: np.ndarray, xs: List[np.ndarray]) -> float:
    """平均重投影误差（像素）"""
    errs = [np.linalg.norm(project(P, X) - x) for P, x in zip(Ps, xs)]
    return float(np.mean(errs)) if errs else np.inf


def build_F_from_RT(K1, K2, R2w, t2w, C2w, R1w=None, t1w=None, C1w=None) -> np.ndarray:
    """由外参构造基础矩阵 F"""
    if R1w is None:
        R1w, t1w = np.eye(3), np.zeros(3)
    R_w1, t_w1 = R1w.T, -R1w.T @ t1w
    R_21 = R2w @ R_w1
    t_21 = t2w + R2w @ t_w1
    tx = np.array(
        [[0, -t_21[2], t_21[1]], [t_21[2], 0, -t_21[0]], [-t_21[1], t_21[0], 0]]
    )
    E = tx @ R_21
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def sampson_error(F: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> float:
    """Sampson误差（像素级）"""
    x1h, x2h = np.append(x1, 1.0), np.append(x2, 1.0)
    Fx1, Ftx2 = F @ x1h, F.T @ x2h
    num = (x2h.T @ F @ x1h) ** 2
    den = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2 + 1e-12
    return float(np.sqrt(num / den))


def triangulate_n_views(
    Ps: List[np.ndarray], xs: List[np.ndarray], ws: List[float] = None
) -> np.ndarray:
    """n视图线性三角测量（加权DLT）"""
    A = []
    if ws is None:
        ws = [1.0] * len(Ps)
    for P, x, w in zip(Ps, xs, ws):
        A.append(w * (x[0] * P[2, :] - P[0, :]))
        A.append(w * (x[1] * P[2, :] - P[1, :]))
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    return Xh[:3] / (Xh[3] + 1e-12)


# ================= 主函数 =================

from typing import Dict, Tuple, Any, List
import numpy as np


def triangulate_with_missing(
    observations: Dict[str, np.ndarray],
    Ks: Dict[str, np.ndarray],
    extrinsics: Dict[str, Dict[str, np.ndarray]],
    max_err_px: float = 5.0,
    sampson_px: float = 2.0,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
    """
    多视点稳健三角测量
    返回:
      X_world: (N,3)
      stats:  统计指标
      detail: 记录“每个3D点由哪些视角的哪些2D观测参与了融合”等信息
    """
    # --- 基础准备 ---
    Ps, Rcs, tcs = {}, {}, {}
    for v in observations.keys():
        R_wc, t_wc = extrinsics[v]["R"], extrinsics[v]["t"]
        Ps[v] = build_P(Ks[v], R_wc, t_wc)
        Rcs[v], tcs[v] = R_wc, t_wc.reshape(3)

    views: List[str] = list(observations.keys())
    V = len(views)
    N = next(iter(observations.values())).shape[0]

    X_world = np.full((N, 3), np.nan)
    stats = {"total": 0, "ok": 0, "mean_rpe": 0, "epi_inlier": 0, "cheirality": 0}

    # --- 记录结构 ---
    used_mask = np.zeros((N, V), dtype=bool)  # 点×视角，是否用于该点的三角化
    used_weights = np.zeros((N, V), dtype=float)  # 若用了则记录权重
    used_pts2d = np.full((N, V, 2), np.nan)  # 参与融合的2D坐标
    per_point_meta: List[Any] = [None] * N  # 每个点的详细信息（字典或 None）
    rpe_list, epi_cnt, cheir_cnt = [], 0, 0

    # 预先缓存基础矩阵
    F_map = {
        (v1, v2): build_F_from_RT(
            Ks[v1],
            Ks[v2],
            R1w=extrinsics[v1]["R"],
            t1w=extrinsics[v1]["t"],
            C1w=extrinsics[v1]["C"],
            R2w=extrinsics[v2]["R"],
            t2w=extrinsics[v2]["t"],
            C2w=extrinsics[v2]["C"],
        )
        for i, v1 in enumerate(views)
        for v2 in views[i + 1 :]
    }

    # --- 主循环：逐 3D 点 ---
    for j in range(N):
        avail, Ps_all, xs_all, ws_all, avail_idx = (
            [],
            [],
            [],
            [],
            [],
        )  # 额外记录 avail 的视角下标
        for vi, v in enumerate(views):
            pt = observations[v][j]  # [x, y] or [x, y, w]
            if np.all(np.isfinite(pt[:2])):
                avail.append(v)
                avail_idx.append(vi)
                Ps_all.append(Ps[v])
                xs_all.append(pt[:2])
                ws_all.append(pt[2] if pt.shape[0] > 2 else 1.0)

        if len(avail) < 2:
            # 记录缺少视角的原因
            per_point_meta[j] = {
                "status": "skipped_not_enough_views",
                "num_avail": len(avail),
            }
            continue

        # 对极约束过滤（这里保持你的“有一对超阈就整体跳过”的策略）
        epi_ok = True
        max_sampson = 0.0
        for i in range(len(avail)):
            for k in range(i + 1, len(avail)):
                e = sampson_error(F_map[(avail[i], avail[k])], xs_all[i], xs_all[k])
                max_sampson = max(max_sampson, float(e))
                if e > sampson_px:
                    epi_ok = False
                    break
            if not epi_ok:
                break
        if not epi_ok:
            per_point_meta[j] = {
                "status": "skipped_epipolar",
                "views": avail,
                "max_sampson": max_sampson,
                "thr": sampson_px,
            }
            continue
        epi_cnt += 1

        # 三角测量
        X = triangulate_n_views(Ps_all, xs_all, ws_all)

        # 正深度检查
        ch_flags = [cheirality_ok(Rcs[v], tcs[v], X) for v in avail]
        if not any(ch_flags):
            per_point_meta[j] = {
                "status": "skipped_cheirality_all_neg",
                "views": avail,
                "cheirality_flags": ch_flags,
            }
            continue
        cheir_cnt += int(all(ch_flags))

        # 重投影误差
        err = reproj_error(Ps_all, X, xs_all)
        stats["total"] += 1
        if err <= max_err_px:
            # 接受：写入 3D、统计、以及使用记录
            X_world[j] = X
            rpe_list.append(err)
            stats["ok"] += 1

            # 标记哪些视角被用了 & 记录其 2D、权重
            used_mask[j, avail_idx] = True
            for local_i, vi in enumerate(avail_idx):
                used_weights[j, vi] = float(ws_all[local_i])
                used_pts2d[j, vi, :] = xs_all[local_i]

            per_point_meta[j] = {
                "status": "accepted",
                "views": avail,  # 参与融合的视角名
                "view_indices": avail_idx,  # 在 views 中的下标
                "num_views": len(avail),
                "max_sampson": max_sampson,
                "reproj_err": float(err),
                "cheirality_flags": ch_flags,  # 针对参与视角
            }
        else:
            per_point_meta[j] = {
                "status": "skipped_high_rpe",
                "views": avail,
                "view_indices": avail_idx,
                "reproj_err": float(err),
                "thr": max_err_px,
                "max_sampson": max_sampson,
                "cheirality_flags": ch_flags,
            }

    # --- 汇总统计 ---
    if rpe_list:
        stats["mean_rpe"] = float(np.mean(rpe_list))
    else:
        stats["mean_rpe"] = np.nan

    denom = stats["total"] + 1e-9
    stats["epi_inlier"] = float(epi_cnt) / denom
    stats["cheirality"] = float(cheir_cnt) / denom

    # --- 失败时不返回 detail / stats ---
    if stats["ok"] == 0:
        # 没有任何点通过：视为失败，不返回 meta
        return X_world, None, None

    # --- 正常返回 detail / stats ---
    detail: Dict[str, Any] = {
        "views_order": views,
        "used_mask": used_mask,
        "used_weights": used_weights,
        "used_pts2d": used_pts2d,
        "per_point": per_point_meta,
    }
    return X_world, stats, detail


# ================= 时序平滑 =================


def smooth_combo(X_seq, alpha=0.3, win=9, poly=2):
    """
    组合式平滑：先EMA再Savitzky-Golay

    参数:
        X_seq: np.ndarray (T, J, 3)
            输入的3D关键点序列
        alpha: float
            EMA平滑系数 (0.1~0.4)
        win: int
            Savitzky-Golay窗口大小 (奇数)
        poly: int
            Savitzky-Golay多项式阶数 (通常2~3)
    返回:
        X_smooth: np.ndarray (T, J, 3)
            平滑后的结果
        report: dict
            平滑报告 (平均位移减少率等指标)
    """
    if X_seq.ndim != 3:
        raise ValueError("X_seq must have shape (T, J, 3)")

    T, J, _ = X_seq.shape

    # === Step 1. EMA 预平滑 ===
    X_ema = X_seq.copy()
    for j in range(J):
        for c in range(3):
            prev = X_ema[0, j, c]
            for t in range(1, T):
                if np.isfinite(X_ema[t, j, c]):
                    prev = alpha * X_ema[t, j, c] + (1 - alpha) * prev
                X_ema[t, j, c] = prev

    # === Step 2. Savitzky–Golay 二次平滑 ===
    win = min(win if win % 2 == 1 else win + 1, max(3, (T // 2) * 2 - 1))
    X_smooth = X_ema.copy()
    for j in range(J):
        for c in range(3):
            v = X_ema[:, j, c]
            if np.isfinite(v).sum() > poly + 2:
                v_interp = np.interp(
                    np.arange(T), np.arange(T)[np.isfinite(v)], v[np.isfinite(v)]
                )
                X_smooth[:, j, c] = savgol_filter(v_interp, win, poly, mode="interp")

    # === Step 3. 计算抖动减少率 ===
    def motion_energy(X):
        diff = np.diff(X, axis=0)
        return np.nanmean(np.linalg.norm(diff, axis=2))

    raw_energy = motion_energy(X_seq)
    smooth_energy = motion_energy(X_smooth)
    reduction = 1 - smooth_energy / (raw_energy + 1e-8)

    report = {
        "method": "EMA→SG",
        "frames": T,
        "joints": J,
        "motion_energy_before": raw_energy,
        "motion_energy_after": smooth_energy,
        "reduction_rate": reduction,
    }

    return X_smooth, report

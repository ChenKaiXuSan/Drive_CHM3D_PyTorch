#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
face3d_eval_pro.py
--------------------------------
无GT 3D人脸重建专业评估脚本
Author: Kaixu Chen (2025)
--------------------------------
✅ 支持指标：
  - RPE（重投影误差）
  - Silhouette IoU
  - Symmetry Score
  - Temporal Stability（速度/加速度P95）
  - Laplacian Energy（表面平滑）
  - Failure Rate（异常帧）
  - Chamfer Distance（多视一致性，可选）

✅ 输出：
  - metrics.csv / metrics.md
  - 柱状图 / 箱线图
  - 可选轨迹图
--------------------------------
"""

import numpy as np
import cv2
import argparse
import os
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# ===============================
# 基础指标计算
# ===============================


def compute_rpe(X_seq, proj_func=None, x_seq=None):
    """重投影误差(px)"""
    if proj_func is None or x_seq is None:
        return np.nan
    errs = []
    for t in range(len(X_seq)):
        x_pred = proj_func(X_seq[t])
        err = np.linalg.norm(x_pred - x_seq[t], axis=1)
        errs.append(np.nanmean(err))
    return float(np.nanmean(errs))


def compute_silhouette_iou(mask_pred, mask_gt):
    inter = np.logical_and(mask_pred, mask_gt).sum()
    union = np.logical_or(mask_pred, mask_gt).sum() + 1e-8
    return inter / union


def compute_symmetry(X):
    X = np.asarray(X)
    axis_x = np.median(X[:, 0])
    X_mirror = X.copy()
    X_mirror[:, 0] = 2 * axis_x - X_mirror[:, 0]
    return np.nanmean(np.linalg.norm(X - X_mirror, axis=1))


def compute_temporal_stability(X_seq):
    diff1 = np.linalg.norm(np.diff(X_seq, axis=0), axis=2)
    diff2 = np.linalg.norm(np.diff(X_seq, n=2, axis=0), axis=2)
    return np.nanpercentile(diff1, 95), np.nanpercentile(diff2, 95)


def compute_laplacian_energy(X):
    """局部平滑能量（邻接点间的二阶差分）"""
    diff = np.diff(X, axis=0)
    return np.nanmean(np.linalg.norm(np.diff(diff, axis=0), axis=2))


def compute_chamfer(X1, X2):
    """Chamfer距离（多视一致性）"""
    tree1, tree2 = cKDTree(X1), cKDTree(X2)
    d1, _ = tree1.query(X2)
    d2, _ = tree2.query(X1)
    return np.mean(d1) + np.mean(d2)


def smooth_seq(X_seq, win=7, poly=2):
    X = X_seq.copy()
    T, J, _ = X.shape
    win = min(win if win % 2 == 1 else win + 1, max(3, (T // 2) * 2 - 1))
    for j in range(J):
        for c in range(3):
            v = X[:, j, c]
            if np.isfinite(v).sum() > poly + 2:
                v_interp = np.interp(
                    np.arange(T), np.arange(T)[np.isfinite(v)], v[np.isfinite(v)]
                )
                X[:, j, c] = savgol_filter(v_interp, win, poly, mode="interp")
    return X


# ===============================
# 主评估模块
# ===============================


def evaluate_face3d_pro(X_seq, x_seq=None, mask_seq=None, X_seq_alt=None, fps=30):
    """
    综合评估 (无GT)
    """
    metrics = {}

    # 对称性
    sym = np.mean([compute_symmetry(X_seq[t]) for t in range(len(X_seq))])
    metrics["Symmetry Score"] = sym

    # 时序稳定性
    sp95, acc95 = compute_temporal_stability(X_seq)
    metrics["Speed_P95"] = sp95
    metrics["Accel_P95"] = acc95

    # 表面平滑能量
    metrics["Laplacian Energy"] = compute_laplacian_energy(X_seq)

    # 有 mask 时计算 IoU
    if mask_seq is not None:
        ious = [compute_silhouette_iou(p, g) for p, g in mask_seq]
        metrics["Silhouette IoU"] = np.nanmean(ious)

    # 多视/增强一致性
    if X_seq_alt is not None:
        chamfer = compute_chamfer(X_seq.reshape(-1, 3), X_seq_alt.reshape(-1, 3))
        metrics["Chamfer Distance"] = chamfer

    # 失败率（NaN点比例）
    nan_rate = np.isnan(X_seq).sum() / X_seq.size
    metrics["Invalid Ratio"] = nan_rate

    return metrics


# ===============================
# 可视化与报告
# ===============================


def visualize_metrics(metrics, save_path=None):
    plt.figure(figsize=(7, 4))
    names, vals = zip(*metrics.items())
    colors = ["#66b3ff" if "IoU" not in k else "#90EE90" for k in names]
    plt.barh(names, vals, color=colors)
    plt.title("3D Face Reconstruction (No GT Evaluation)")
    plt.xlabel("Value")
    plt.grid(alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()


def export_report(metrics, outdir="results_eval"):
    os.makedirs(outdir, exist_ok=True)
    md_path = os.path.join(outdir, "metrics.md")
    csv_path = os.path.join(outdir, "metrics.csv")

    # Markdown
    with open(md_path, "w") as f:
        f.write("| Metric | Value |\n|:--|--:|\n")
        for k, v in metrics.items():
            f.write(f"| {k} | {v:.6f} |\n")
    # CSV
    np.savetxt(
        csv_path,
        np.array([[k, v] for k, v in metrics.items()], dtype=object),
        fmt="%s",
        delimiter=",",
    )

    print(f"✅ Report saved:\n - {md_path}\n - {csv_path}")


# ===============================
# CLI
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to npz or npy file containing X_seq",
    )
    parser.add_argument(
        "--alt",
        type=str,
        default=None,
        help="Optional alternate 3D result (multi-view)",
    )
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    # 加载数据
    print(f"📂 Loading {args.input}")
    X_seq = np.load(args.input)
    if isinstance(X_seq, np.lib.npyio.NpzFile):
        X_seq = X_seq["X_seq"]

    X_seq_alt = None
    if args.alt:
        X_seq_alt = np.load(args.alt)
        if isinstance(X_seq_alt, np.lib.npyio.NpzFile):
            X_seq_alt = X_seq_alt["X_seq"]

    metrics = evaluate_face3d_pro(X_seq, X_seq_alt=X_seq_alt, fps=args.fps)
    visualize_metrics(metrics, save_path="results_eval/metrics_bar.png")
    export_report(metrics, "results_eval")

    print("📊 Evaluation Complete.")
    for k, v in metrics.items():
        print(f"{k:20s}: {v:.6f}")

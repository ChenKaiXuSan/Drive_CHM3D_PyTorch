#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
pose_visualization.py (refactored+optimized)
--------------------------------
3D Mesh 与相机位姿可视化工具（解耦 & 稳健版）
- 支持多相机绘制与多视角导出
- 使用 Line3DCollection 加速连线
- 自动等比例边界
- NaN过滤、安全索引
- 支持颜色映射、索引、Y轴翻转
- 可选导出 default/front/left/right 四个视角
--------------------------------
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from pathlib import Path
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import mediapipe as mp

logger = logging.getLogger(__name__)
mp_face_mesh = mp.solutions.face_mesh
TES = np.array(list(mp_face_mesh.FACEMESH_TESSELATION))
CON = np.array(list(mp_face_mesh.FACEMESH_CONTOURS))

# ================= Helper Functions =================


def _calc_view_from_point(
    target: np.ndarray, cam_pos: np.ndarray
) -> Tuple[float, float]:
    v = np.asarray(cam_pos, float).reshape(3) - np.asarray(target, float).reshape(3)
    vx, vy, vz = v[0], v[1], v[2]
    if np.allclose([vx, vy, vz], 0, atol=1e-12):
        return 20.0, -60.0
    azim = float(np.degrees(np.arctan2(vy, vx)))
    elev = float(np.degrees(np.arctan2(vz, np.hypot(vx, vy))))
    return elev, azim


def _gather_cam_centers(rt_info: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    centers = {}
    if not rt_info:
        return centers
    for name, ext in rt_info.items():
        try:
            R_wc = np.asarray(ext["R"], float).reshape(3, 3)
            t_wc = np.asarray(ext["t"], float).reshape(3)
            Cw = -R_wc.T @ t_wc
            centers[str(name)] = Cw
        except Exception:
            continue
    return centers


def _save_views(
    fig: plt.Figure,
    ax: plt.Axes,
    base_path: Path,
    look_at: np.ndarray,
    cam_centers: Dict[str, np.ndarray],
    default_elev_azim: Tuple[float, float],
    tags: List[str],
    dpi: int = 220,
) -> Dict[str, Path]:
    out_paths: Dict[str, Path] = {}
    base_path = Path(base_path)
    stem = base_path.stem
    parent = base_path.parent
    parent.mkdir(parents=True, exist_ok=True)

    for tag in tags:
        if tag == "default":
            elev, azim = default_elev_azim
        else:
            if tag not in cam_centers:
                continue
            elev, azim = _calc_view_from_point(look_at, cam_centers[tag])
        ax.view_init(elev=elev, azim=azim)
        p = parent / f"{stem}_{tag}.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        out_paths[tag] = p
    return out_paths


def _save_mesh_views(fig, ax, base_path, default_elev_azim, dpi=220):
    # 以人脸朝向为准，保存预设视角
    # 前：人脸面向相机
    presets = {
        "default": default_elev_azim,
        "left": (0.0, 180.0),
        "front": (0.0, 90.0),
        "back": (0.0, -90.0),
        "top": (90.0, 0.0),
        "bottom": (-90.0, 0.0),
    }

    out = {}
    base_path = Path(base_path)
    stem, parent = base_path.stem, base_path.parent
    for tag in presets:
        save_path = parent / tag / f"{stem}_.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        elev, azim = presets[tag]
        ax.view_init(elev=elev, azim=azim)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        out[tag] = save_path
    return out


# ================= Core Drawing =================


def draw_camera(
    ax: plt.Axes,
    R_wc: np.ndarray,
    t_wc: np.ndarray,
    K: Optional[np.ndarray] = None,
    image_size: Optional[Tuple[int, int]] = None,
    axis_len: float = 0.05,
    frustum_depth: float = 0.1,
    color_axes: Tuple[str, str, str] = ("r", "g", "b"),
    frustum_alpha: float = 0.4,
    label: Optional[str] = None,
    invK: Optional[np.ndarray] = None,
) -> np.ndarray:
    R_wc = np.asarray(R_wc, float).reshape(3, 3)
    t_wc = np.asarray(t_wc, float).reshape(3)
    Cw = -R_wc.T @ t_wc

    for vec, col in zip(R_wc, color_axes):
        ax.plot(
            [Cw[0], Cw[0] + vec[0] * axis_len],
            [Cw[1], Cw[1] + vec[1] * axis_len],
            [Cw[2], Cw[2] + vec[2] * axis_len],
            c=col,
            lw=1.2,
        )

    if (K is not None or invK is not None) and image_size is not None:
        if invK is None:
            invK = np.linalg.inv(np.asarray(K, float).reshape(3, 3))
        W, H = map(float, image_size)
        corners_px = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], float).T
        rays = invK @ corners_px
        scale = frustum_depth / np.clip(rays[2, :], 1e-6, None)
        rays *= scale
        corners_w = (R_wc.T @ rays).T + Cw  # 正确：cam→world

        for p in corners_w:
            ax.plot(
                [Cw[0], p[0]],
                [Cw[1], p[1]],
                [Cw[2], p[2]],
                c="k",
                lw=0.6,
                alpha=frustum_alpha,
            )
        loop = [0, 1, 2, 3, 0]
        ax.plot(
            corners_w[loop, 0],
            corners_w[loop, 1],
            corners_w[loop, 2],
            c="k",
            lw=0.6,
            alpha=frustum_alpha,
        )

    if label:
        ax.text(Cw[0], Cw[1], Cw[2], label, fontsize=8, color="k")
    return Cw


# ================= Main Visualization =================


def visualize_3d_mesh(
    mesh_3d: np.ndarray,
    save_path: Path,
    title: str = "Triangulated 3D Mesh",
    rt_info: Optional[Dict[str, Any]] = None,
    K: Optional[Dict[str, np.ndarray]] = None,
    image_size: Optional[Tuple[int, int]] = None,
    point_size: int = 8,
    color_points: Union[str, np.ndarray] = "dodgerblue",
    draw_tessellation: bool = True,
    draw_contours: bool = False,
    show_indices: bool = False,
    invert_y: bool = False,
    dpi: int = 220,
    save_views: Optional[List[str]] = None,
    default_view: Tuple[float, float] = (20.0, -60.0),
) -> Dict[str, Path]:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if mesh_3d.ndim != 2 or mesh_3d.shape[1] != 3:
        raise ValueError(f"mesh_3d 应为 (N,3)，得到 {mesh_3d.shape}")

    mesh = np.asarray(mesh_3d, float)
    if not np.isfinite(mesh).all():
        mask = np.isfinite(mesh).all(axis=1)
        mesh = mesh[mask]
        logger.warning("Mesh 含 NaN，已过滤失效点。")

    if invert_y:
        mesh = mesh.copy()
        mesh[:, 1] *= -1.0

    if (
        isinstance(color_points, np.ndarray)
        and color_points.ndim == 1
        and color_points.shape[0] == mesh.shape[0]
    ):
        vals = color_points.astype(float)
        vmin, vmax = np.nanpercentile(vals, [2, 98])
        vals = np.clip((vals - vmin) / max(vmax - vmin, 1e-9), 0, 1)
        cmap = plt.get_cmap("viridis")
        colors = cmap(vals)
    else:
        colors = color_points

    invK_map = {
        name: np.linalg.inv(Km) for name, Km in (K or {}).items() if Km is not None
    }

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (up)")

    ax.scatter(
        mesh[:, 0], mesh[:, 1], mesh[:, 2], c=colors, s=point_size, depthshade=True
    )

    segs = []
    if draw_tessellation:
        idx_ok = (TES[:, 0] < len(mesh)) & (TES[:, 1] < len(mesh))
        if np.any(idx_ok):
            segs.append(np.stack([mesh[TES[idx_ok, 0]], mesh[TES[idx_ok, 1]]], axis=1))
    if draw_contours:
        idx_ok = (CON[:, 0] < len(mesh)) & (CON[:, 1] < len(mesh))
        if np.any(idx_ok):
            segs.append(np.stack([mesh[CON[idx_ok, 0]], mesh[CON[idx_ok, 1]]], axis=1))
    if len(segs) > 0:
        segs = np.concatenate(segs, axis=0)
        lc = Line3DCollection(segs, colors="gray", linewidths=0.3, alpha=0.6)
        ax.add_collection3d(lc)

    if show_indices and len(mesh) <= 600:
        for i, (x, y, z) in enumerate(mesh):
            ax.text(x, y, z, str(i), size=7, color="k")

    if rt_info:
        for name, ext in rt_info.items():
            draw_camera(
                ax,
                ext["R"],
                ext["t"],
                K=(K or {}).get(name),
                image_size=image_size,
                axis_len=0.08,
                frustum_depth=0.25,
                label=str(name),
                invK=invK_map.get(name),
            )

    out_paths: Dict[str, Path] = {"main": save_path}

    if save_views:
        extras = _save_mesh_views(
            fig=fig,
            ax=ax,
            base_path=save_path,
            default_elev_azim=default_view,
            dpi=dpi,
        )
        out_paths.update(extras)

    plt.close(fig)
    logger.info(f"Saved 3D mesh visualization → {save_path}")
    return out_paths

import os
from typing import Dict, Tuple, Optional, Iterable, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------- Visualization -------------------------


def _make_axes_points(ext, axis_len: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    Xc = np.eye(3) * axis_len  # cam axes endpoints in cam
    # Xc[:,1] *= -1                      # 这里渲染的时候让 Y 轴反向更直观
    Xw = (ext.R_cw @ Xc) + ext.t_cw.reshape(3, 1)
    return Xw, ext.C


def _frustum_corners_world(
    ext: Any,
    K: Optional[np.ndarray],
    img_size: Optional[Tuple[int, int]],
    depth: float,
) -> np.ndarray:
    """
    返回 (3,4) 的世界坐标四角点。img_size = (width, height)
    """
    if K is None or img_size is None:
        fov = np.deg2rad(60 / 2)
        w = depth * np.tan(fov)
        h = w * 0.75
        corners_cam = np.array(
            [[-w, -h, depth], [w, -h, depth], [w, h, depth], [-w, h, depth]], float
        ).T
    else:
        W, H = img_size
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pixels = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], float)
        x = (pixels[:, 0] - cx) * (depth / fx)
        y = (pixels[:, 1] - cy) * (depth / fy)
        z = np.full_like(x, depth)
        corners_cam = np.vstack([x, y, z])

    corners_world = (ext.R_cw @ corners_cam) + ext.t_cw.reshape(3, 1)
    return corners_world


def draw_cameras_matplotlib(
    extrinsics_map: Dict[int, Any],
    K_map: Optional[Dict[int, np.ndarray]] = None,
    img_size: Dict[str, Tuple[int, int]] = None,
    frustum_depth: float = 0.4,
    axis_len: float = 0.2,
    figsize: Tuple[int, int] = (8, 8),
    elev: float = 20,
    azim: float = 40,
    auto_equal: bool = True,
    save_path: Optional[str] = None,
):
    """画相机中心、坐标轴、视锥；img_size=(width,height)"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    all_pts = []

    for cid, ext in sorted(extrinsics_map.items()):
        # center
        ax.scatter(ext.C[0], ext.C[1], ext.C[2], marker="o")
        all_pts.append(ext.C)

        # axes
        axes_end, Cw = _make_axes_points(ext, axis_len=axis_len)
        ax.plot(
            [Cw[0], axes_end[0, 0]],
            [Cw[1], axes_end[1, 0]],
            [Cw[2], axes_end[2, 0]],
            color="r",
        )  # X
        ax.plot(
            [Cw[0], axes_end[0, 1]],
            [Cw[1], axes_end[1, 1]],
            [Cw[2], axes_end[2, 1]],
            color="g",
        )  # Y
        ax.plot(
            [Cw[0], axes_end[0, 2]],
            [Cw[1], axes_end[1, 2]],
            [Cw[2], axes_end[2, 2]],
            color="b",
        )  # Z

        # frustum
        K = None if K_map is None else K_map.get(cid, None)
        corners = _frustum_corners_world(ext, K, img_size, depth=frustum_depth)  # (3,4)

        order = [0, 1, 2, 3, 0]  # rim
        ax.plot(corners[0, order], corners[1, order], corners[2, order], color="orange")
        for j in range(4):  # rays
            ax.plot(
                [Cw[0], corners[0, j]],
                [Cw[1], corners[1, j]],
                [Cw[2], corners[2, j]],
                color="orange",
            )

        all_pts.append(corners.T)
        ax.text(ext.C[0], ext.C[1], ext.C[2], f"Cam {cid}", fontsize=9)

    all_pts = np.vstack(all_pts)
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_zlabel("Z (world)")
    ax.view_init(elev=elev, azim=azim)

    if auto_equal and all_pts.size > 0:
        mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
        centers = (mins + maxs) / 2.0
        ranges = maxs - mins
        radius = float(np.max(ranges)) * 0.6 if np.all(ranges > 0) else 1.0
        ax.set_xlim([centers[0] - radius, centers[0] + radius])
        ax.set_ylim([centers[1] - radius, centers[1] + radius])
        ax.set_zlim([centers[2] - radius, centers[2] + radius])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=180)
    return fig, ax


# ----------------------- Multi-view export -----------------------
def save_multi_views(
    extrinsics_map: Dict[int, Any],
    K_map: Optional[Dict[int, np.ndarray]] = None,
    img_size: Optional[Dict[str, Tuple[int, int]]] = None,
    save_prefix: str = "cameras",
    views: Optional[Dict[str, Dict[str, float]]] = None,
):
    """
    一次性导出多视图（可自定义）
    默认：front/left/top
    """
    if views is None:
        views = {
            "front": dict(elev=0, azim=0),  # 从 +X 看
            "left": dict(elev=0, azim=90),  # 从 +Y 看
            "top": dict(elev=90, azim=-90),  # 俯视
        }
    for name, view in views.items():
        fig, _ = draw_cameras_matplotlib(
            extrinsics_map,
            K_map=K_map,
            img_size=img_size,
            elev=view["elev"],
            azim=view["azim"],
            save_path=f"{save_prefix}_{name}.png",
        )
        plt.close(fig)

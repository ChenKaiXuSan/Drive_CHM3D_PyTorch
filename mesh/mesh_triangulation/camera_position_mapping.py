#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable, Union, Any
import numpy as np
from pathlib import Path

from mesh_triangulation.vis.camera_position_visualization import (
    draw_cameras_matplotlib,
    save_multi_views,
)

# ---------------------------- Core types ----------------------------


@dataclass
class Extrinsics:
    """
    OpenCV 约定：
      X_cam = R_wc * X_world + t_wc
    同时提供 camera->world：
      X_world = R_cw * X_cam + t_cw，且 t_cw == C (相机中心, world)
    """

    R_wc: np.ndarray  # (3,3)
    t_wc: np.ndarray  # (3,)
    R_cw: np.ndarray  # (3,3)
    t_cw: np.ndarray  # (3,)
    C: np.ndarray  # (3,)


# ---------------------------- Math utils ----------------------------


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Zero-length vector cannot be normalized")
    return v / n


def _proj_to_so3(R: np.ndarray) -> np.ndarray:
    """将 3x3 矩阵投影到最近的 SO(3)，稳定正交化。"""
    U, _, Vt = np.linalg.svd(R)
    R_ = U @ Vt
    if np.linalg.det(R_) < 0:
        U[:, -1] *= -1
        R_ = U @ Vt
    return R_


def rodrigues_to_R(rvec: Iterable[float]) -> np.ndarray:
    """Rodrigues 向量(弧度, 世界->相机) -> 旋转矩阵(3x3)"""
    r = np.asarray(rvec, float).reshape(3)
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], float)
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return _proj_to_so3(R)


def ypr_deg_to_R(
    yaw_deg: float, pitch_deg: float, roll_deg: float, order: str = "ZYX"
) -> np.ndarray:
    """
    yaw/pitch/roll(度) -> R_cw(相机->世界)；默认 ZYX：R = Rz(yaw)*Ry(pitch)*Rx(roll)
    """
    y, p, r = np.deg2rad([yaw_deg, pitch_deg, roll_deg])
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], float)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], float)
    mapping = {"X": Rx, "Y": Ry, "Z": Rz}
    R = np.eye(3)
    for ax in order:
        R = mapping[ax] @ R
    return _proj_to_so3(R)


# ------------------------- Pose construction ------------------------


def lookat_Rt(
    C: Iterable[float], T: Iterable[float], up: Iterable[float] = (0, 0, 1), filp_y=True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    由相机位置 C 和目标点 T 生成世界->相机 (R_wc, t_wc)
    相机坐标：+Z 前、+X 右、+Y 下 (OpenCV)
    """
    C = np.asarray(C, float).reshape(3)
    T = np.asarray(T, float).reshape(3)
    up = _normalize(np.asarray(up, float).reshape(3))

    z_cam = T - C
    if np.linalg.norm(z_cam) < 1e-12:
        raise ValueError("Camera and target coincide")
    z_cam = _normalize(z_cam)

    x_cam = np.cross(z_cam, up)
    if np.linalg.norm(x_cam) < 1e-6:  # up ~ z_cam，换一个应急 up
        alt_up = np.array([0, 1, 0]) if abs(z_cam[1]) < 0.9 else np.array([1, 0, 0])
        x_cam = np.cross(z_cam, alt_up)
    x_cam = _normalize(x_cam)

    y_cam = np.cross(x_cam, z_cam)
    y_cam = _normalize(y_cam)
    if filp_y:
        y_cam = -y_cam

    R_cw = np.stack([x_cam, y_cam, z_cam], axis=1)  # cam->world
    R_wc = R_cw.T
    t_wc = -R_wc @ C
    return R_wc, t_wc


def compose_extrinsics_from(
    cam_pos: Iterable[float],
    orientation: Dict[str, Union[Tuple[float, float, float], float]],
    orientation_mode: str = "lookat",
    up: Iterable[float] = (0, 0, 1),
) -> Extrinsics:
    """
    orientation_mode:
      - 'lookat': orientation['target'] = (x,y,z)
      - 'ypr'   : orientation['yaw','pitch','roll'] (度)，得到 R_cw 再转置
      - 'rodrigues': orientation['rvec'] 世界->相机
    """
    C = np.asarray(cam_pos, float).reshape(3)

    if orientation_mode == "lookat":
        R_wc, t_wc = lookat_Rt(C, orientation["target"], up=up)
    elif orientation_mode == "ypr":
        R_cw = ypr_deg_to_R(
            orientation["yaw"], orientation["pitch"], orientation["roll"], order="ZYX"
        )
        R_wc = R_cw.T
        t_wc = -R_wc @ C
    elif orientation_mode == "rodrigues":
        R_wc = rodrigues_to_R(orientation["rvec"])
        t_wc = -R_wc @ C
    else:
        raise ValueError(f"Unknown orientation_mode: {orientation_mode}")

    R_cw = R_wc.T
    t_cw = C
    return Extrinsics(R_wc=R_wc, t_wc=t_wc, R_cw=R_cw, t_cw=t_cw, C=C)


def build_extrinsics_map(
    camera_layout: Dict[str, Dict[str, any]], default_up: Iterable[float] = (0, 0, 1)
) -> Dict[int, Extrinsics]:
    """
    camera_layout[id] = {
        'pos': (x,y,z),
        one of: 'target' | 'ypr' | 'rvec',
        optional: 'up'
    }
    """
    out: Dict[int, Extrinsics] = {}
    for cid, spec in camera_layout.items():
        pos = spec["pos"]
        if "target" in spec:
            ext = compose_extrinsics_from(
                pos,
                {"target": spec["target"]},
                orientation_mode="lookat",
                up=spec.get("up", default_up),
            )
        elif "ypr" in spec:
            yaw, pitch, roll = spec["ypr"]
            ext = compose_extrinsics_from(
                pos,
                {"yaw": yaw, "pitch": pitch, "roll": roll},
                orientation_mode="ypr",
                up=spec.get("up", default_up),
            )
        elif "rvec" in spec:
            ext = compose_extrinsics_from(
                pos,
                {"rvec": spec["rvec"]},
                orientation_mode="rodrigues",
                up=spec.get("up", default_up),
            )
        else:
            raise ValueError(
                f"Camera {cid} must contain one of: 'target' | 'ypr' | 'rvec'"
            )
        out[cid] = ext
    return out


def make_projection_matrices(
    extrinsics_map: Dict[int, Extrinsics], K_map: Optional[Dict[int, np.ndarray]] = None
) -> Dict[int, np.ndarray]:
    """P = K [R|t]；若 K_map=None 则用 I3。"""
    P_map: Dict[int, np.ndarray] = {}
    for cid, ext in extrinsics_map.items():
        K = np.eye(3) if K_map is None else K_map[cid]
        Rt = np.hstack([ext.R_wc, ext.t_wc.reshape(3, 1)])
        P_map[cid] = K @ Rt
    return P_map


# ------------------------- Layout helpers ---------------------------


def angles_to_position(
    target: Tuple[float, float, float],
    distance: float,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    z_override: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
    由 (distance, yaw, pitch) 计算相机中心 C，使相机 +Z 指向 target。
    约定：Z上,yaw 绕 Z轴，（逆时针为 +），pitch 仰角。
    注意，这里的相机是需要看向 target 的。

    Args:
        target (Tuple[float, float, float]): 目标点坐标
        distance (float): 相机与目标点的距离
        yaw_deg (float): 相机绕 Z 轴旋转的角度
        pitch_deg (float, optional): 相机绕 X 轴旋转的角度。默认为 0.0。
        z_override (Optional[float], optional): 覆盖相机 Z 轴坐标。默认为 None。

    Returns:
        Tuple[float, float, float]: 相机中心 C 的坐标
    """
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    f = np.array(
        [np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)], float
    )  # cam forward(+Z)
    T = np.asarray(target, float)
    C = T - distance * f
    if z_override is not None:
        C[2] = z_override
    return tuple(C.tolist())


# ----------------------- Camera intrinsics utils -----------------------


def resize_K(
    K: np.ndarray,
    old_size: tuple[int, int],
    new_size: tuple[int, int],
    mode: str = "letterbox",
) -> np.ndarray:
    """
    根据分辨率变化调整相机内参矩阵 K。

    参数
    ----
    K : np.ndarray (3x3)
        原始内参矩阵。
    old_size : (W, H)
        原始图像分辨率（像素）。
    new_size : (W', H')
        目标图像分辨率（像素）。
    mode : str
        缩放模式，可选：
          - "non_uniform"：宽高分别缩放（可能改变比例）
          - "letterbox"：保持比例，缩小并填充边
          - "center_crop"：保持比例，放大并裁剪居中部分

    返回
    ----
    K_new : np.ndarray (3x3)
        新的内参矩阵。
    """

    W, H = old_size
    Wn, Hn = new_size

    if mode == "non_uniform":
        sx, sy = Wn / W, Hn / H
        tx, ty = 0.0, 0.0

    elif mode == "letterbox":
        s = min(Wn / W, Hn / H)
        sx = sy = s
        tx = (Wn - s * W) / 2.0
        ty = (Hn - s * H) / 2.0

    elif mode == "center_crop":
        s = max(Wn / W, Hn / H)
        sx = sy = s
        tx = -(s * W - Wn) / 2.0
        ty = -(s * H - Hn) / 2.0

    else:
        raise ValueError(
            "mode must be one of: 'non_uniform', 'letterbox', 'center_crop'"
        )

    # 变换矩阵
    A = np.array([[sx, 0, tx], [0, sy, ty], [0, 0, 1]], dtype=float)

    return A @ K


def prepare_camera_position(
    K: Dict[str, np.array],
    T: Tuple[float, float, float],
    z: float,
    output_path: Optional[str] = None,
    img_size: Optional[Tuple[int, int]] = None,
    dist_front: float = 0.62,
    dist_left: float = 0.85,
    dist_right: float = 0.85,
    baseline: float = 0.70,
) -> Dict[str, Dict]:
    """
    准备相机位置数据，返回字典格式，包含相机ID、位置和朝向信息。

    Args:
        extrinsics_map (Dict[int, Extrinsics]): 包含相机外参的字典，键为相机ID，值为Extrinsics对象。

    Returns:
        Dict[int, Dict]: 包含相机位置和朝向信息的字典，键为相机ID，值为包含位置和朝向的字典。
    """

    output_path = Path(output_path)

    CAMERA_LAYOUT: Dict[str, Dict[str, Any]] = {}

    # 计算左右相机的坐标
    x_half = baseline / 2
    y_side = math.sqrt(dist_left**2 - x_half**2)  # ≈ 0.7746

    CAMERA_LAYOUT = {
        "front": {
            "pos": (0.0, dist_front, z),
            "target": T,
            "yaw": 180.0,  # 正前方相机 → 朝向目标
            "raw_K": np.array(K["front"]).reshape(3, 3),
        },
        "left": {
            "pos": (-x_half, y_side, z),
            "target": T,
            "yaw": math.degrees(math.atan2(0.0 - y_side, 0.0 - (-x_half))),  # ≈ -115.5°
            "raw_K": np.array(K["left"]).reshape(3, 3),
        },
        "right": {
            "pos": (x_half, y_side, z),
            "target": T,
            "yaw": math.degrees(math.atan2(0.0 - y_side, 0.0 - x_half)),  # ≈ -65.5°
            "raw_K": np.array(K["right"]).reshape(3, 3),
        },
    }

    extr_map = build_extrinsics_map(CAMERA_LAYOUT)

    # —— 打印外参摘要 ——
    for cid, ext in sorted(extr_map.items()):
        print(f"[Cam {cid}]")
        print(
            "R_wc=\n",
            np.array2string(ext.R_wc, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print(
            "t_wc=",
            np.array2string(ext.t_wc, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print(
            "C(w) =",
            np.array2string(ext.C, formatter={"float_kind": lambda x: f"{x: .5f}"}),
        )
        print()

    rt_info = dict()
    for cid, ext in extr_map.items():
        rt_info[cid] = {
            "R": ext.R_wc,
            "t": ext.t_wc,
            "C": ext.C,
        }

    # —— 可视化 & 多视图导出 ——
    K_resized_map = {
        cid: resize_K(
            param["raw_K"],
            old_size=(2304, 1296),
            new_size=(332, 225),
            mode="letterbox",
        )
        for cid, param in CAMERA_LAYOUT.items()
    }

    for cid in CAMERA_LAYOUT.keys():
        CAMERA_LAYOUT[cid]["resized_K"] = K_resized_map[cid]

    K_raw_map = {
        cid: param["raw_K"] for cid, param in CAMERA_LAYOUT.items() if "raw_K" in param
    }

    draw_cameras_matplotlib(
        extr_map,
        K_map=K_raw_map,
        img_size=(2304, 1296),
        frustum_depth=0.5,
        axis_len=0.25,
        save_path=output_path / "camera_position" / "original_camera_poses.png",
    )

    save_multi_views(
        extr_map,
        K_map=K_raw_map,
        img_size=(2304, 1296),
        save_prefix=output_path / "camera_position" / "original_camera_poses",
    )

    draw_cameras_matplotlib(
        extr_map,
        K_map=K_resized_map,
        img_size=(332, 225),
        frustum_depth=0.5,
        axis_len=0.25,
        save_path=output_path / "camera_position" / "resized_camera_poses.png",
    )
    save_multi_views(
        extr_map,
        K_map=K_resized_map,
        img_size=(332, 225),
        save_prefix=output_path / "camera_position" / "resized_camera_poses",
    )

    return {
        "layout": CAMERA_LAYOUT,  # {cid: {"pos": (x, y, z), "target": T}}
        "extrinsics_map": extr_map,  # {cid: Extrinsics}
        "K_map": K_resized_map,  # {cid: K}
        "rt_info": rt_info,  # {cid: {"R": R_wc, "t": t_wc, "C": C}}
    }


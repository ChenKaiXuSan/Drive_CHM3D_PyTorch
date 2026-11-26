#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/head_movement_analysis/angle_calculator.py
Project: /workspace/code/head_movement_analysis
Created Date: Saturday February 7th 2026
Author: Kaixu Chen
-----
Comment: 头部转角计算模块
Have a good code time :)
-----
Copyright (c) 2026 The University of Tsukuba
-----
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# MHR70关键点索引定义
KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "neck": 69,
}

# 标注方向到角度方向的映射 (pitch_dir, yaw_dir)
# 前、上、下、左、右 + 4 个斜向；左右均以驾驶员自身为准
LABEL_DIRECTION_MAP = {
    "front": (0, 0),  # 朝向正前方
    "up": (1, 0),  # 向上点头
    "down": (-1, 0),  # 向下点头
    "left": (0, -1),  # 向左转头
    "right": (0, 1),  # 向右转头
    "left_up": (1, -1),
    "left_down": (-1, -1),
    "right_up": (1, 1),
    "right_down": (-1, 1),
}

# 坐标约定：关键点在相机坐标系(OpenCV)，X 向图像右、Y 向下、Z 远离相机。
# 驾驶员面对相机时朝向 -Z，驾驶员的右侧是图像左侧(-X)，上方是 -Y。


def wrap_angle_deg(angle):
    """把角度(标量或数组)折回 [-180, 180)。"""
    return (np.asarray(angle, dtype=np.float64) + 180.0) % 360.0 - 180.0


def circular_mean_deg(angles) -> float:
    """角度平均值（在 ±180° 处也正确）。"""
    rad = np.radians(np.asarray(angles, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())))


def estimate_front_yaw(yaw_vals: np.ndarray, half_width: float = 3.0) -> float:
    """估计"正面"的 yaw：最常见的头部朝向（1° 直方图众数，±half_width 内取均值）。

    不能假设正面就是 0°：侧面相机里正面约在 ±30~40°，front 相机也有几度偏差。
    """
    yaw = wrap_angle_deg(np.asarray(yaw_vals, dtype=np.float64))
    yaw = yaw[np.isfinite(yaw)]
    if yaw.size == 0:
        return 0.0
    hist, edges = np.histogram(yaw, bins=np.arange(-180.0, 181.0, 1.0))
    # 环形平滑，避免众数落在单个噪声格子上
    smooth = np.convolve(np.r_[hist[-2:], hist, hist[:2]], np.ones(5), mode="valid")
    center = float(edges[int(np.argmax(smooth))] + 0.5)
    near = wrap_angle_deg(yaw - center)
    near = near[np.abs(near) <= half_width]
    offset = float(near.mean()) if near.size else 0.0
    return float(wrap_angle_deg(center + offset))


def axis_direction_matched(match: Dict) -> bool:
    """一个标注涉及的所有轴都匹配才算方向正确（斜向需要左右和上下都对）。"""
    active = [
        bool(match.get(f"{axis}_match", False))
        for axis in ("yaw", "pitch")
        if match.get(f"{axis}_axis_active", False)
    ]
    return bool(active) and all(active)


def extract_head_keypoints(
    keypoints_3d: np.ndarray, keypoint_indices: Dict[str, int]
) -> Optional[Dict[str, np.ndarray]]:
    """
    从3D关键点数组中提取头部相关的关键点

    Args:
        keypoints_3d: 形状为 (70, 3) 的3D关键点数组
        keypoint_indices: 关键点名称到索引的映射字典

    Returns:
        包含头部关键点的字典，如果关键点无效则返回None
    """
    head_kpts = {}
    required_kpts = [
        "nose",
        "left_eye",
        "right_eye",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "neck",
    ]

    for name in required_kpts:
        idx = keypoint_indices[name]
        kpt = keypoints_3d[idx]

        # 检查关键点是否有效
        if not np.isfinite(kpt).all():
            logger.warning(f"Invalid keypoint: {name} (index {idx})")
            return None

        head_kpts[name] = kpt

    return head_kpts


def calculate_head_angles(
    head_kpts: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    """
    计算头部的两个转动角度

    Args:
        head_kpts: 包含头部关键点的字典

    Returns:
        (pitch, yaw) 两个角度，单位为度
        - pitch: 俯仰角（上下点头），正值表示抬头，负值表示低头
        - yaw: 偏航角（左右转头），正值表示向右转，负值表示向左转
    """
    # ===== 1. 计算Pitch（俯仰角）=====
    # Use nose-ears plane geometry: ear-center->nose vector with ear-axis
    # component removed, then measure elevation against the horizontal X-Z plane.
    pitch_deg = calculate_pitch_from_nose_ears_plane(head_kpts)
    if not np.isfinite(pitch_deg):
        pitch_deg = 0.0

    # ===== 2. 计算Yaw（偏航角）=====
    yaw_deg = calculate_yaw_from_nose_ears(head_kpts)

    return pitch_deg, yaw_deg


def calculate_yaw_from_nose_ears(head_kpts: Dict[str, np.ndarray]) -> float:
    """Yaw of the ear-center -> nose direction in the horizontal (X-Z) plane.

    Returns:
        yaw in degrees; 0 = facing the camera, driver's right > 0, left < 0.
    """
    nose = np.asarray(head_kpts["nose"], dtype=np.float64)
    left_ear = np.asarray(head_kpts["left_ear"], dtype=np.float64)
    right_ear = np.asarray(head_kpts["right_ear"], dtype=np.float64)

    if (
        np.any(np.isnan(nose))
        or np.any(np.isnan(left_ear))
        or np.any(np.isnan(right_ear))
    ):
        return float("nan")

    forward = nose - (left_ear + right_ear) / 2.0
    if np.hypot(forward[0], forward[2]) < 1e-6:
        return float("nan")

    # 面对相机时 forward 指向 -Z；驾驶员右侧是图像左侧(-X)
    return float(np.degrees(np.arctan2(-forward[0], -forward[2])))


def calculate_pitch_from_nose_ears_plane(
    head_kpts: Dict[str, np.ndarray],
) -> float:
    """Calculate pitch from the nose-ears geometry on a single frame.

    Steps:
    1) Compute ear center C and vector v = nose - C
    2) Remove ear-axis (left-ear -> right-ear) component from v
    3) Measure elevation angle of the residual vector against the horizontal
       (X-Z) plane; the camera Y axis points down, so up is -Y

    Returns:
        pitch in degrees, where up > 0 and down < 0
    """
    nose = np.asarray(head_kpts["nose"], dtype=np.float64)
    left_ear = np.asarray(head_kpts["left_ear"], dtype=np.float64)
    right_ear = np.asarray(head_kpts["right_ear"], dtype=np.float64)

    if (
        np.any(np.isnan(nose))
        or np.any(np.isnan(left_ear))
        or np.any(np.isnan(right_ear))
    ):
        return float("nan")

    ear_center = (left_ear + right_ear) / 2.0
    ear_axis = right_ear - left_ear
    ear_norm = np.linalg.norm(ear_axis)
    if ear_norm < 1e-6:
        return float("nan")

    ear_axis = ear_axis / ear_norm
    v = nose - ear_center
    v_perp = v - np.dot(v, ear_axis) * ear_axis

    horiz = np.hypot(v_perp[0], v_perp[2])
    if horiz < 1e-6 and abs(v_perp[1]) < 1e-6:
        return float("nan")

    return float(np.degrees(np.arctan2(-v_perp[1], horiz)))


def estimate_stable_front_baseline(
    raw_pitch_vals: np.ndarray,
    yaw_vals: np.ndarray,
    front_ratio: float = 0.15,
    min_front_frames: int = 30,
    candidate_expand: float = 3.0,
    mad_k: float = 2.5,
    max_iters: int = 5,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate a robust front baseline from frame-wise pitch/yaw sequences.

    The baseline is estimated in three stages:
    1) Select candidate frames whose yaw is closest to the most common yaw
       (:func:`estimate_front_yaw`)
    2) Iteratively reject outliers in candidate pitch by median + MAD
    3) Choose the closest inliers to the robust center and take median pitch

    Returns:
        baseline_pitch, selected_front_indices, yaw_candidate_indices
    """
    pitch_arr = np.asarray(raw_pitch_vals, dtype=np.float64)
    yaw_arr = np.asarray(yaw_vals, dtype=np.float64)

    valid_mask = np.isfinite(pitch_arr) & np.isfinite(yaw_arr)
    if not np.any(valid_mask):
        return 0.0, np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    valid_indices = np.where(valid_mask)[0]
    pitch_valid = pitch_arr[valid_indices]
    yaw_valid = yaw_arr[valid_indices]
    yaw_offset = np.abs(wrap_angle_deg(yaw_valid - estimate_front_yaw(yaw_valid)))

    n_frames = len(valid_indices)
    n_front = max(min_front_frames, int(n_frames * front_ratio))
    n_front = min(n_front, n_frames)

    n_candidates = min(n_frames, max(n_front, int(n_front * candidate_expand)))
    candidate_order = np.argsort(yaw_offset)[:n_candidates]
    candidate_local_idx = candidate_order
    candidate_idx = valid_indices[candidate_local_idx]
    candidate_pitch = pitch_valid[candidate_local_idx]

    center = float(np.median(candidate_pitch))
    inlier_mask = np.ones(candidate_pitch.shape[0], dtype=bool)

    for _ in range(max_iters):
        working = candidate_pitch[inlier_mask]
        if working.size == 0:
            break

        new_center = float(np.median(working))
        mad = float(np.median(np.abs(working - new_center)))
        robust_sigma = max(1e-6, 1.4826 * mad)
        new_inlier_mask = np.abs(candidate_pitch - new_center) <= mad_k * robust_sigma

        if np.all(new_inlier_mask == inlier_mask):
            center = new_center
            inlier_mask = new_inlier_mask
            break

        center = new_center
        inlier_mask = new_inlier_mask

    inlier_idx = candidate_idx[inlier_mask]

    if inlier_idx.size < min(10, n_front):
        selected_local = np.argsort(yaw_offset)[:n_front]
        selected_idx = valid_indices[selected_local]
    else:
        inlier_pitch = pitch_arr[inlier_idx]
        selected_order = np.argsort(np.abs(inlier_pitch - center))
        selected_idx = inlier_idx[selected_order[: min(n_front, inlier_idx.size)]]

    baseline = (
        float(np.median(pitch_arr[selected_idx])) if selected_idx.size > 0 else 0.0
    )
    return baseline, selected_idx.astype(np.int64), candidate_idx.astype(np.int64)


def direction_match(angle_value: float, expected_dir: int, threshold: float) -> bool:
    """
    检查角度值是否匹配预期方向

    Args:
        angle_value: 计算出的角度值（度）
        expected_dir: 期望方向 (1=正向, -1=负向, 0=中立)
        threshold: 阈值（度）

    Returns:
        bool: 是否匹配
    """
    if expected_dir > 0:
        return angle_value > threshold
    if expected_dir < 0:
        return angle_value < -threshold
    return abs(angle_value) <= threshold


def classify_label(pitch: float, yaw: float, threshold: float) -> str:
    """根据 pitch/yaw 与阈值的关系输出离散方向标签。"""
    pitch_dir = 0
    yaw_dir = 0

    if pitch > threshold:
        pitch_dir = 1
    elif pitch < -threshold:
        pitch_dir = -1

    if yaw > threshold:
        yaw_dir = 1
    elif yaw < -threshold:
        yaw_dir = -1

    for label, (expected_pitch, expected_yaw) in LABEL_DIRECTION_MAP.items():
        if expected_pitch == pitch_dir and expected_yaw == yaw_dir:
            return label

    return "front"

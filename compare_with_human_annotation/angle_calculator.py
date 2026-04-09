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

# 标注方向到角度方向的映射
# 5个基本方向：前、上、下、左、右
LABEL_DIRECTION_MAP = {
    "front": (0, 0),  # 朝向正前方
    "up": (1, 0),  # 向上点头
    "down": (-1, 0),  # 向下点头
    "left": (0, -1),  # 向左转头
    "right": (0, 1),  # 向右转头
}


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
    # component removed, then measure elevation against XY plane.
    pitch_deg = calculate_pitch_from_nose_ears_plane(head_kpts)
    if not np.isfinite(pitch_deg):
        pitch_deg = 0.0

    # ===== 2. 计算Yaw（偏航角）=====
    # Use the ear-center -> eye-center -> nose line projected onto XY plane,
    # then measure the signed angle against +Y axis.
    nose = head_kpts["nose"]
    left_eye = head_kpts["left_eye"]
    right_eye = head_kpts["right_eye"]
    left_ear = head_kpts["left_ear"]
    right_ear = head_kpts["right_ear"]

    eye_center = (left_eye + right_eye) / 2.0
    ear_center = (left_ear + right_ear) / 2.0

    # This combines the two connected segments ear->eye and eye->nose.
    forward_line = (eye_center - ear_center) + (nose - eye_center)
    forward_xy = np.array([forward_line[0], forward_line[1]], dtype=np.float64)
    if np.linalg.norm(forward_xy) < 1e-6:
        yaw_deg = 0.0
    else:
        yaw = np.arctan2(forward_xy[0], forward_xy[1])
        yaw_deg = np.degrees(yaw)

    return pitch_deg, yaw_deg


def calculate_pitch_from_nose_ears_plane(
    head_kpts: Dict[str, np.ndarray],
) -> float:
    """Calculate pitch from the nose-ears geometry on a single frame.

    Steps:
    1) Compute ear center C and vector v = nose - C
    2) Remove ear-axis (left-ear -> right-ear) component from v
    3) Measure elevation angle of the residual vector against XY plane

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

    horiz = np.linalg.norm(v_perp[:2])
    if horiz < 1e-6 and abs(v_perp[2]) < 1e-6:
        return float("nan")

    return float(np.degrees(np.arctan2(v_perp[2], horiz)))


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
    1) Select candidate frames with smallest |yaw|
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

    n_frames = len(valid_indices)
    n_front = max(min_front_frames, int(n_frames * front_ratio))
    n_front = min(n_front, n_frames)

    n_candidates = min(n_frames, max(n_front, int(n_front * candidate_expand)))
    candidate_order = np.argsort(np.abs(yaw_valid))[:n_candidates]
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
        selected_local = np.argsort(np.abs(yaw_valid))[:n_front]
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

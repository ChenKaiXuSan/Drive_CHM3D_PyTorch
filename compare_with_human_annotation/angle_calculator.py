#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
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
'''
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
    "front": (0, 0),   # 朝向正前方
    "up": (1, 0),      # 向上点头
    "down": (-1, 0),   # 向下点头
    "left": (0, -1),   # 向左转头
    "right": (0, 1),   # 向右转头
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


def _normalize_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    """Normalize a vector and return None for degenerate inputs."""
    norm = np.linalg.norm(vector)
    if norm < 1e-6:
        return None
    return vector / norm


def _build_head_axes(
    head_kpts: Dict[str, np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build an orthonormal head coordinate system from keypoints."""
    nose = head_kpts["nose"]
    left_eye = head_kpts["left_eye"]
    right_eye = head_kpts["right_eye"]
    left_ear = head_kpts["left_ear"]
    right_ear = head_kpts["right_ear"]
    left_shoulder = head_kpts["left_shoulder"]
    right_shoulder = head_kpts["right_shoulder"]
    neck = head_kpts["neck"]

    eye_center = (left_eye + right_eye) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0

    # Head left-right axis: combine eye line and ear line for stability.
    left_right_raw = (right_eye - left_eye) + (right_ear - left_ear)
    x_axis = _normalize_vector(left_right_raw)
    if x_axis is None:
        return None

    # Head forward axis: primarily from facial geometry, stabilized by neck.
    forward_raw = nose - eye_center
    if np.linalg.norm(forward_raw) < 1e-6:
        forward_raw = nose - neck
    if np.linalg.norm(forward_raw) < 1e-6:
        forward_raw = nose - shoulder_center

    # Remove any component along the left-right axis before normalizing.
    forward_raw = forward_raw - np.dot(forward_raw, x_axis) * x_axis
    z_axis = _normalize_vector(forward_raw)
    if z_axis is None:
        return None

    # Make the forward axis point roughly from the face toward the camera/front.
    forward_hint = nose - eye_center
    if np.dot(z_axis, forward_hint) < 0:
        z_axis = -z_axis

    # Reconstruct the up axis to keep the basis orthogonal and stable.
    y_axis = _normalize_vector(np.cross(z_axis, x_axis))
    if y_axis is None:
        return None

    return x_axis, y_axis, z_axis


def calculate_head_angles(
    head_kpts: Dict[str, np.ndarray],
) -> Tuple[float, float]:
    """
    计算头部的三个转动角度

    Args:
        head_kpts: 包含头部关键点的字典

    Returns:
        (pitch, yaw) 两个角度，单位为度
        - pitch: 俯仰角（上下点头），正值表示抬头，负值表示低头
        - yaw: 偏航角（左右转头），正值表示向右转，负值表示向左转
    """
    axes = _build_head_axes(head_kpts)
    if axes is None:
        logger.warning("Failed to build head coordinate system")
        return 0.0, 0.0

    x_axis, y_axis, z_axis = axes

    # ===== 1. 计算Pitch（俯仰角）=====
    # Use the forward axis elevation in camera coordinates.
    pitch = np.arctan2(z_axis[1], np.sqrt(z_axis[0] ** 2 + z_axis[2] ** 2))
    pitch_deg = np.degrees(pitch)

    # ===== 2. 计算Yaw（偏航角）=====
    # Use the forward axis horizontal deviation.
    yaw = np.arctan2(z_axis[0], -z_axis[2])
    yaw_deg = np.degrees(yaw)

    return pitch_deg, yaw_deg


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
    """
    根据Pitch和Yaw角度分类标注标签

    Args:
        pitch: 俯仰角（度）
        yaw: 偏航角（度）
        threshold: 分类阈值（度）

    Returns:
        str: 分类后的标签名称
    """
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

    if pitch_dir == 0 and yaw_dir == 0:
        return "front"

    for label, (expected_pitch, expected_yaw) in LABEL_DIRECTION_MAP.items():
        if expected_pitch == pitch_dir and expected_yaw == yaw_dir:
            return label

    return "front"

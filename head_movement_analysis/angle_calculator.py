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

from compare_with_human_annotation.angle_calculator import (
    calculate_pitch_from_nose_ears_plane,
    calculate_yaw_from_nose_ears,
    circular_mean_deg,
    estimate_stable_front_baseline,
    wrap_angle_deg,
)

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
LABEL_DIRECTION_MAP = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "left_up": (1, -1),
    "left_down": (-1, -1),
    "right_up": (1, 1),
    "right_down": (-1, 1),
    "front": (0, 0),
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
) -> Tuple[float, float, float]:
    """
    计算头部的三个转动角度

    Args:
        head_kpts: 包含头部关键点的字典

    Returns:
        (pitch, yaw, roll) 三个角度，单位为度
        - pitch: 俯仰角（上下点头），正值表示抬头，负值表示低头
        - yaw: 偏航角（左右转头），正值表示向右转，负值表示向左转
        - roll: 翻滚角（左右倾斜），正值表示向右倾斜，负值表示向左倾斜
    """
    # Pitch/Yaw 与 compare_with_human_annotation 使用同一套公式：
    # 相机坐标系 X 向图像右、Y 向下、Z 远离相机；右转为正、抬头为正。
    pitch_deg = calculate_pitch_from_nose_ears_plane(head_kpts)
    yaw_deg = calculate_yaw_from_nose_ears(head_kpts)

    # Roll：两眼连线与水平面的夹角，头正时为 0，向驾驶员右侧倾斜为正
    # （驾驶员左眼在图像右侧，所以用 左眼 - 右眼）
    eye_vec = head_kpts["left_eye"] - head_kpts["right_eye"]
    roll_deg = float(np.degrees(np.arctan2(-eye_vec[1], eye_vec[0])))

    return pitch_deg, yaw_deg, roll_deg


def subtract_front_baseline(
    results: Dict[int, Dict[str, float]],
) -> Dict[int, Dict[str, float]]:
    """把整段序列的角度减去"正面"基线（最常见朝向），使正面注视约为 0°。"""
    frames = sorted(results)
    if not frames:
        return results
    pitch = np.array([results[f]["pitch"] for f in frames], dtype=np.float64)
    yaw = np.array([results[f]["yaw"] for f in frames], dtype=np.float64)
    baseline_pitch, selected, _ = estimate_stable_front_baseline(pitch, yaw)
    baseline_yaw = circular_mean_deg(yaw[selected]) if selected.size else 0.0
    return {
        f: {
            **results[f],
            "pitch": float(results[f]["pitch"] - baseline_pitch),
            "yaw": float(wrap_angle_deg(results[f]["yaw"] - baseline_yaw)),
        }
        for f in frames
    }


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

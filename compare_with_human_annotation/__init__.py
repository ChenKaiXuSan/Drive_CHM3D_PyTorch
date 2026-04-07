#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
与人工标注比较的头部姿态分析模块

该模块用于从融合后的3D关键点数据中分析头部的转动角度，并与人工标注进行比较。

核心功能:
- 读取融合后的3D关键点数据
- 计算头部的Pitch（俯仰角）、Yaw（偏航角）、Roll（翻滚角）
- 加载头部动作标注
- 比较计算角度与标注，计算匹配率

使用示例:
    >>> from compare_with_human_annotation import HeadPoseAnalyzer
    >>> from compare_with_human_annotation import load_head_movement_annotations
    >>> from pathlib import Path
    >>> 
    >>> # 加载标注
    >>> annotations = load_head_movement_annotations(
    ...     Path("/workspace/data/annotation/label/full.json")
    ... )
    >>> 
    >>> # 创建分析器
    >>> analyzer = HeadPoseAnalyzer(annotation_dict=annotations)
    >>> 
    >>> # 分析并比较
    >>> results = analyzer.analyze_sequence_with_annotations(
    ...     video_id="01_day_high",
    ...     fused_dir=Path("/workspace/data/head3d_fuse_results/01/昼多い/fused_npz")
    ... )
    >>> 
    >>> # 查看比较结果
    >>> for frame_idx, comparison in results["comparisons"].items():
    ...     print(f"Frame {frame_idx}: {comparison['matches']}")

Author: Kaixu Chen (chenkaixusan@gmail.com)
Date: February 7, 2026
"""

from .load import (
    HeadMovementLabel,
    get_all_annotations_for_frame,
    get_annotation_for_frame,
    load_fused_keypoints,
    load_sam3d_keypoints,
    load_head_movement_annotations,
    load_majority_voted_annotations,
    load_annotations_by_annotator,
    load_multi_annotator_annotations,
)
from .angle_calculator import (
    KEYPOINT_INDICES,
    LABEL_DIRECTION_MAP,
    calculate_head_angles,
    direction_match,
    extract_head_keypoints,
)
__version__ = "1.0.0"
__author__ = "Kaixu Chen"
__email__ = "chenkaixusan@gmail.com"


def __getattr__(name):
    """懒加载 HeadPoseAnalyzer，避免运行 main 模块时重复导入警告。"""
    if name == "HeadPoseAnalyzer":
        from .main import HeadPoseAnalyzer

        return HeadPoseAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "HeadPoseAnalyzer",
    "KEYPOINT_INDICES",
    "LABEL_DIRECTION_MAP",
    "HeadMovementLabel",
    "load_fused_keypoints",
    "load_sam3d_keypoints",
    "load_head_movement_annotations",
    "load_multi_annotator_annotations",
    "get_annotation_for_frame",
    "calculate_head_angles",
    "classify_label",
    "direction_match",
    "extract_head_keypoints",
    "get_all_annotations_for_frame",
]

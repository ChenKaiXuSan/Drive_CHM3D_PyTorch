#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/head_movement_analysis/load.py
Project: /workspace/code/head_movement_analysis
Created Date: Saturday February 7th 2026
Author: Kaixu Chen

用于加载3D关键点数据和头部动作标注的模块
"""
import dataclasses
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 标注帧号(24 fps) -> 相机帧号(29.97 fps)
from common.frame_rate import annotation_to_camera_frame

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class HeadMovementLabel:
    """头部动作标注信息"""
    start_frame: int
    end_frame: int
    label: str  # e.g., "right", "left", "up", "down"
    
    def contains_frame(self, frame_idx: int) -> bool:
        """检查帧索引是否在标注范围内"""
        return self.start_frame <= frame_idx <= self.end_frame


def load_fused_keypoints(npy_path: Path) -> Optional[np.ndarray]:
    """
    读取融合后的3D关键点数据

    Args:
        npy_path: .npy文件路径

    Returns:
        形状为 (70, 3) 的3D关键点数组，如果读取失败则返回None
    """
    try:
        data = np.load(npy_path, allow_pickle=True).item()
        keypoints_3d = data.get("fused_keypoints_3d", None)

        if keypoints_3d is None:
            logger.error(f"No 'fused_keypoints_3d' found in {npy_path}")
            return None

        # 确保关键点数据形状正确
        if keypoints_3d.ndim == 3:
            keypoints_3d = keypoints_3d[0]  # 去除batch维度

        if keypoints_3d.shape[0] < 70:
            logger.error(
                f"Expected at least 70 keypoints, got {keypoints_3d.shape[0]}"
            )
            return None

        return keypoints_3d[:70]  # 只保留前70个关键点

    except Exception as e:
        logger.error(f"Failed to load keypoints from {npy_path}: {e}")
        return None


def load_head_movement_annotations(json_path: Path) -> Dict[str, List[HeadMovementLabel]]:
    """
    从JSON文件加载头部动作标注
    
    Args:
        json_path: 标注JSON文件路径
        
    Returns:
        字典，键为视频ID，值为HeadMovementLabel列表
        例如: {"person_01_day_high": [HeadMovementLabel(...), ...]}
    """
    if not json_path.exists():
        logger.error(f"Annotation file not found: {json_path}")
        return {}
    
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load annotation file {json_path}: {e}")
        return {}
    
    annotations = {}
    
    for item in data:
        # 从annotations列表中获取result
        if "annotations" not in item or not item["annotations"]:
            continue
            
        annotation = item["annotations"][0]  # 取第一个标注
        result = annotation.get("result", [])
        
        # 提取视频路径信息 (从data字段中获取)
        data_field = item.get("data", {})
        video_path = data_field.get("video", "")
        if not video_path:
            continue
            
        # 从文件名提取person_id和env_id
        # 例如: "person_01_day_high_1.mp4" -> "01_day_high"
        file_name = os.path.basename(video_path)
        parts = file_name.replace(".mp4", "").split("_")
        
        if len(parts) < 4:
            continue
        
        # 构建视频ID (例如: "01_day_high")
        video_id = f"{parts[1]}_{parts[2]}_{parts[3]}"  # 01_day_high
        
        labels = []
        
        # 解析每个timeline标注
        for annotation_item in result:
            if annotation_item.get("type") != "timelinelabels":
                continue
                
            value = annotation_item.get("value", {})
            ranges = value.get("ranges", [])
            timelinelabels = value.get("timelinelabels", [])
            
            if not ranges or not timelinelabels:
                continue
            
            # 每个range对应一个标注
            for range_item in ranges:
                start = range_item.get("start")
                end = range_item.get("end")
                
                if start is None or end is None:
                    continue
                
                # 每个range可能有多个label
                for label in timelinelabels:
                    labels.append(HeadMovementLabel(
                        start_frame=annotation_to_camera_frame(start),
                        end_frame=annotation_to_camera_frame(end),
                        label=label
                    ))
        
        if labels:
            annotations[video_id] = labels
            logger.debug(f"Loaded {len(labels)} annotations for {video_id}")
    
    logger.info(f"Loaded annotations for {len(annotations)} videos")
    return annotations


def load_multi_annotator_annotations(
    json_path: Path,
) -> Tuple[Optional[str], List[List[HeadMovementLabel]]]:
    """
    加载multi_view_driver_action格式的多标注者标注

    Args:
        json_path: 标注JSON文件路径 (person_XX_day_high_h265.json)

    Returns:
        (video_id, labels_by_annotator)
        video_id 例如: "03_day_high"
        labels_by_annotator: 每个标注者的HeadMovementLabel列表
    """
    if not json_path.exists():
        logger.error(f"Annotation file not found: {json_path}")
        return None, []

    try:
        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        logger.error(f"Failed to load annotation file {json_path}: {exc}")
        return None, []

    if not isinstance(data, dict):
        logger.error(f"Unexpected annotation format: {json_path}")
        return None, []

    video_file = data.get("video_file", "")
    if not video_file:
        video_file = os.path.basename(json_path)

    file_name = os.path.basename(video_file)
    base_name = file_name.replace(".mp4", "").replace(".json", "")
    parts = base_name.split("_")
    if len(parts) < 4:
        logger.error(f"Unable to parse video id from: {file_name}")
        return None, []

    video_id = f"{parts[1]}_{parts[2]}_{parts[3]}"  # 03_day_high

    annotations = data.get("annotations", [])
    labels_by_annotator: List[List[HeadMovementLabel]] = []

    for annotator in annotations:
        labels: List[HeadMovementLabel] = []
        for item in annotator.get("videoLabels", []):
            ranges = item.get("ranges", [])
            timelinelabels = item.get("timelinelabels", [])

            if not ranges or not timelinelabels:
                continue

            for range_item in ranges:
                start = range_item.get("start")
                end = range_item.get("end")
                if start is None or end is None:
                    continue
                for label in timelinelabels:
                    labels.append(
                        HeadMovementLabel(
                            start_frame=annotation_to_camera_frame(start),
                            end_frame=annotation_to_camera_frame(end),
                            label=label,
                        )
                    )

        labels_by_annotator.append(labels)

    if labels_by_annotator:
        logger.info(
            "Loaded %d annotators from %s",
            len(labels_by_annotator),
            json_path.name,
        )
    else:
        logger.warning(f"No labels found in {json_path}")

    return video_id, labels_by_annotator


def get_annotation_for_frame(
    annotations: List[HeadMovementLabel],
    frame_idx: int
) -> Optional[HeadMovementLabel]:
    """
    获取指定帧的标注信息
    
    Args:
        annotations: HeadMovementLabel列表
        frame_idx: 帧索引
        
    Returns:
        如果该帧有标注，返回HeadMovementLabel，否则返回None
    """
    for annotation in annotations:
        if annotation.contains_frame(frame_idx):
            return annotation
    return None


def get_all_annotations_for_frame(
    annotations: List[HeadMovementLabel],
    frame_idx: int
) -> List[HeadMovementLabel]:
    """
    获取指定帧的所有标注信息（可能有多个重叠的标注）
    
    Args:
        annotations: HeadMovementLabel列表
        frame_idx: 帧索引
        
    Returns:
        该帧的所有标注列表
    """
    result = []
    for annotation in annotations:
        if annotation.contains_frame(frame_idx):
            result.append(annotation)
    return result

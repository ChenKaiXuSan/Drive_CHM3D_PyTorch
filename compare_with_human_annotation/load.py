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
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

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


def _extract_video_id(video_path: str) -> Optional[str]:
    """从视频路径中提取 video_id。"""
    if not video_path:
        return None

    file_name = os.path.basename(video_path)
    parts = file_name.replace(".mp4", "").split("_")
    if len(parts) < 4:
        return None

    return f"{parts[1]}_{parts[2]}_{parts[3]}"


def _extract_labels_from_result(result: List[dict]) -> List[HeadMovementLabel]:
    """从单个标注结果中提取所有时间段标签。"""
    labels: List[HeadMovementLabel] = []

    for annotation_item in result:
        if annotation_item.get("type") != "timelinelabels":
            continue

        value = annotation_item.get("value", {})
        ranges = value.get("ranges", [])
        timelinelabels = value.get("timelinelabels", [])

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
                        start_frame=int(start),
                        end_frame=int(end),
                        label=label,
                    )
                )

    return labels


def _vote_majority_annotations(
    labels_by_annotator: List[List[HeadMovementLabel]],
) -> List[HeadMovementLabel]:
    """对多个标注者的帧级标签执行多数投票，并压缩为连续时间段。"""
    if not labels_by_annotator:
        return []

    all_labels = [label for annotator_labels in labels_by_annotator for label in annotator_labels]
    if not all_labels:
        return []

    min_frame = min(label.start_frame for label in all_labels)
    max_frame = max(label.end_frame for label in all_labels)

    voted_frames: List[Tuple[int, Optional[str]]] = []

    for frame_idx in range(min_frame, max_frame + 1):
        votes: List[str] = []

        for annotator_labels in labels_by_annotator:
            active_labels = {label.label for label in annotator_labels if label.contains_frame(frame_idx)}

            if len(active_labels) == 1:
                votes.append(next(iter(active_labels)))

        if not votes:
            voted_frames.append((frame_idx, None))
            continue

        counter = Counter(votes)
        best_label, best_count = counter.most_common(1)[0]

        # 3位标注者的多数投票需要至少2票同意
        if best_count >= 2:
            voted_frames.append((frame_idx, best_label))
        else:
            voted_frames.append((frame_idx, None))

    compressed_labels: List[HeadMovementLabel] = []
    current_label: Optional[str] = None
    segment_start: Optional[int] = None
    previous_frame: Optional[int] = None

    for frame_idx, label in voted_frames:
        if label is None:
            if current_label is not None and segment_start is not None and previous_frame is not None:
                compressed_labels.append(
                    HeadMovementLabel(
                        start_frame=segment_start,
                        end_frame=previous_frame,
                        label=current_label,
                    )
                )
            current_label = None
            segment_start = None
            previous_frame = frame_idx
            continue

        if current_label is None:
            current_label = label
            segment_start = frame_idx
            previous_frame = frame_idx
            continue

        if label != current_label:
            compressed_labels.append(
                HeadMovementLabel(
                    start_frame=segment_start if segment_start is not None else frame_idx,
                    end_frame=previous_frame if previous_frame is not None else frame_idx,
                    label=current_label,
                )
            )
            current_label = label
            segment_start = frame_idx

        previous_frame = frame_idx

    if current_label is not None and segment_start is not None and previous_frame is not None:
        compressed_labels.append(
            HeadMovementLabel(
                start_frame=segment_start,
                end_frame=previous_frame,
                label=current_label,
            )
        )

    return compressed_labels


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


def load_sam3d_keypoints(npz_path: Path) -> Optional[np.ndarray]:
    """
    读取单视角 SAM3D 的3D关键点数据。

    Args:
        npz_path: *_sam3d_body.npz 文件路径

    Returns:
        形状为 (70, 3) 的3D关键点数组，如果读取失败则返回None
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        if "output" not in data.files:
            logger.error(f"No 'output' found in {npz_path}")
            return None

        output = data["output"].item()
        if not isinstance(output, dict):
            logger.error(f"Unexpected 'output' type in {npz_path}: {type(output)}")
            return None

        keypoints_3d = output.get("pred_keypoints_3d", None)
        if keypoints_3d is None:
            logger.error(f"No 'pred_keypoints_3d' found in {npz_path}")
            return None

        if keypoints_3d.ndim == 3:
            keypoints_3d = keypoints_3d[0]

        if keypoints_3d.shape[0] < 70:
            logger.error(
                f"Expected at least 70 keypoints in {npz_path}, got {keypoints_3d.shape[0]}"
            )
            return None

        return keypoints_3d[:70]

    except Exception as e:
        logger.error(f"Failed to load SAM3D keypoints from {npz_path}: {e}")
        return None


def load_head_movement_annotations(json_path: Path) -> Dict[str, List[HeadMovementLabel]]:
    """
    从JSON文件加载头部动作标注
    
    Args:
        json_path: 标注JSON文件路径
        
    Returns:
        字典，键为视频ID，值为HeadMovementLabel列表
        同一视频下的多个标注者结果会被合并到同一个列表中
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
        annotations_list = item.get("annotations", [])
        if not annotations_list:
            continue
        
        # 提取视频路径信息 (从data字段中获取)
        data_field = item.get("data", {})
        video_path = data_field.get("video", "")
        video_id = _extract_video_id(video_path)
        if not video_id:
            continue
        
        labels = []

        # 解析每个标注者的 timeline 标注
        for annotation in annotations_list:
            labels.extend(_extract_labels_from_result(annotation.get("result", [])))
        
        if labels:
            annotations[video_id] = labels
            logger.debug(f"Loaded {len(labels)} annotations for {video_id}")
    
    logger.info(f"Loaded annotations for {len(annotations)} videos")
    return annotations


def load_majority_voted_annotations(
    json_path: Path,
) -> Dict[str, List[HeadMovementLabel]]:
    """
    从JSON文件加载头部动作标注，并对同一视频下的 3 位标注者做多数投票。

    Returns:
        字典，键为视频ID，值为多数投票后的 HeadMovementLabel 列表。
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

    annotations: Dict[str, List[HeadMovementLabel]] = {}

    for item in data:
        annotations_list = item.get("annotations", [])
        if not annotations_list:
            continue

        data_field = item.get("data", {})
        video_path = data_field.get("video", "")
        video_id = _extract_video_id(video_path)
        if not video_id:
            continue

        labels_by_annotator: List[List[HeadMovementLabel]] = []
        for annotation in annotations_list:
            labels_by_annotator.append(
                _extract_labels_from_result(annotation.get("result", []))
            )

        voted_labels = _vote_majority_annotations(labels_by_annotator)
        if voted_labels:
            annotations[video_id] = voted_labels
            logger.debug(
                "Loaded %d majority-voted annotations for %s",
                len(voted_labels),
                video_id,
            )

    logger.info("Loaded majority-voted annotations for %d videos", len(annotations))
    return annotations


def load_annotations_by_annotator(
    json_path: Path,
) -> Dict[str, List[List[HeadMovementLabel]]]:
    """
    从JSON文件加载头部动作标注，按标注者分组。

    Returns:
        字典，键为视频ID，值为标注者列表，每个标注者是一个HeadMovementLabel列表
        例如: {
            "01_day_high": [
                [Label1_from_annotator1, ...],  # 标注者1的标注
                [Label2_from_annotator2, ...],  # 标注者2的标注
                [Label3_from_annotator3, ...],  # 标注者3的标注
            ]
        }
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

    annotations: Dict[str, List[List[HeadMovementLabel]]] = {}

    for item in data:
        annotations_list = item.get("annotations", [])
        if not annotations_list:
            continue

        data_field = item.get("data", {})
        video_path = data_field.get("video", "")
        video_id = _extract_video_id(video_path)
        if not video_id:
            continue

        labels_by_annotator: List[List[HeadMovementLabel]] = []
        for annotation_idx, annotation in enumerate(annotations_list):
            labels = _extract_labels_from_result(annotation.get("result", []))
            labels_by_annotator.append(labels)
            logger.debug(
                f"Loaded {len(labels)} annotations for {video_id} from annotator {annotation_idx + 1}"
            )

        if labels_by_annotator:
            annotations[video_id] = labels_by_annotator
            logger.debug(f"Loaded {len(labels_by_annotator)} annotators for {video_id}")

    logger.info(f"Loaded annotations (by annotator) for {len(annotations)} videos")
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
                            start_frame=int(start),
                            end_frame=int(end),
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


def get_labeled_frame_ranges(
    annotations: List[HeadMovementLabel],
) -> List[Tuple[int, int]]:
    """Return merged labeled frame ranges from a label list."""
    if not annotations:
        return []

    ranges = sorted((annotation.start_frame, annotation.end_frame) for annotation in annotations)
    merged_ranges: List[Tuple[int, int]] = []

    for start_frame, end_frame in ranges:
        if not merged_ranges:
            merged_ranges.append((start_frame, end_frame))
            continue

        previous_start, previous_end = merged_ranges[-1]
        if start_frame <= previous_end + 1:
            merged_ranges[-1] = (previous_start, max(previous_end, end_frame))
        else:
            merged_ranges.append((start_frame, end_frame))

    return merged_ranges


def get_unlabeled_frame_indices(
    annotations: List[HeadMovementLabel],
    frame_indices: List[int],
) -> List[int]:
    """Return frame indices that are not covered by any annotation interval."""
    if not frame_indices:
        return []

    labeled_ranges = get_labeled_frame_ranges(annotations)
    if not labeled_ranges:
        return list(frame_indices)

    unlabeled_frames: List[int] = []
    range_index = 0
    total_ranges = len(labeled_ranges)

    for frame_idx in frame_indices:
        while range_index < total_ranges and labeled_ranges[range_index][1] < frame_idx:
            range_index += 1

        if range_index < total_ranges:
            start_frame, end_frame = labeled_ranges[range_index]
            if start_frame <= frame_idx <= end_frame:
                continue

        unlabeled_frames.append(frame_idx)

    return unlabeled_frames

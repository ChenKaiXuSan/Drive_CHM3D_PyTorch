import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .load import (
    HeadMovementLabel,
    get_all_annotations_for_frame,
    load_fused_keypoints,
    get_unlabeled_frame_indices,
)
from .angle_calculator import (
    KEYPOINT_INDICES,
    LABEL_DIRECTION_MAP,
    calculate_head_angles,
    estimate_stable_front_baseline,
    direction_match,
    extract_head_keypoints,
)

logger = logging.getLogger(__name__)

class HeadPoseAnalyzer:
    """
    头部姿态分析器 - 与人工标注比较
    从融合后的3D关键点计算头部的三个转动角度，并与标注进行比较。
    
    支持的角度：
    1. Pitch（俯仰角）：上下点头
    2. Yaw（偏航角）：左右转头
    """

    def __init__(self, annotation_dict: Optional[Dict[str, List[HeadMovementLabel]]] = None):
        """初始化头部姿态分析器
        
        Args:
            annotation_dict: 可选的标注字典，用于与计算结果比较
                            字典格式: {video_id: [HeadMovementLabel, ...]}
        """
        self.keypoint_indices = KEYPOINT_INDICES
        self.annotation_dict = annotation_dict

    def analyze_head_pose(self, npy_path: Path) -> Optional[Dict[str, float]]:
        """
        分析单帧的头部姿态

        Args:
            npy_path: 融合后的.npy文件路径

        Returns:
            包含两个角度的字典，如果分析失败则返回None
            {
                'pitch': float,  # 俯仰角（度）
                'yaw': float,    # 偏航角（度）
            }
        """
        # 1. 读取关键点
        keypoints_3d = load_fused_keypoints(npy_path)
        if keypoints_3d is None:
            return None

        # 2. 提取头部关键点
        head_kpts = extract_head_keypoints(keypoints_3d, self.keypoint_indices)
        if head_kpts is None:
            return None

        # 3. 计算角度
        pitch, yaw = calculate_head_angles(head_kpts)

        result = {
            "pitch": pitch,
            "yaw": yaw,
        }
        
        return result

    @staticmethod
    def _apply_baseline(
        angles: Dict[str, float],
        baseline_angles: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """Subtract a baseline from raw angles."""
        if not baseline_angles:
            return dict(angles)

        return {
            "pitch": angles.get("pitch", 0.0) - baseline_angles.get("pitch", 0.0),
            "yaw": angles.get("yaw", 0.0) - baseline_angles.get("yaw", 0.0),
        }

    def analyze_sequence(
        self,
        fused_dir: Path,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> Dict[int, Dict[str, float]]:
        """
        分析一个序列中所有帧的头部姿态

        Args:
            fused_dir: 包含融合npy文件的目录
            start_frame: 起始帧索引（可选）
            end_frame: 结束帧索引（可选）

        Returns:
            字典，键为帧索引，值为包含三个角度的字典
            {
                frame_idx: {
                    'pitch': float,
                    'yaw': float,
                }
            }
        """
        results = {}

        # 获取所有npy文件
        npy_files = sorted(fused_dir.glob("frame_*_fused.npy"))

        for npy_file in npy_files:
            # 从文件名解析帧索引
            frame_idx = int(npy_file.stem.split("_")[1])

            # 检查帧索引是否在指定范围内
            if start_frame is not None and frame_idx < start_frame:
                continue
            if end_frame is not None and frame_idx > end_frame:
                continue

            # 分析该帧
            angles = self.analyze_head_pose(npy_file)
            if angles is not None:
                results[frame_idx] = angles
            else:
                logger.warning(f"Failed to analyze frame {frame_idx}")

        logger.info(f"Successfully analyzed {len(results)} frames")
        return results

    def get_unlabeled_frames(
        self,
        video_id: str,
        fused_dir: Path,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> List[int]:
        """Get frame indices that are not covered by any annotation interval."""
        frame_indices: List[int] = []

        for npy_file in sorted(fused_dir.glob("frame_*_fused.npy")):
            frame_idx = int(npy_file.stem.split("_")[1])
            if start_frame is not None and frame_idx < start_frame:
                continue
            if end_frame is not None and frame_idx > end_frame:
                continue
            frame_indices.append(frame_idx)

        if self.annotation_dict is None or video_id not in self.annotation_dict:
            return frame_indices

        return get_unlabeled_frame_indices(self.annotation_dict[video_id], frame_indices)

    def estimate_front_baseline(
        self,
        video_id: str,
        fused_dir: Path,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        selection_ratio: float = 0.15,
        min_selected_frames: int = 30,
        max_selected_frames: int = 500,
    ) -> Optional[Dict]:
        """Estimate a front baseline from the most front-like unlabeled frames."""
        unlabeled_frames = self.get_unlabeled_frames(
            video_id=video_id,
            fused_dir=fused_dir,
            start_frame=start_frame,
            end_frame=end_frame,
        )

        if not unlabeled_frames:
            return None

        candidate_records: List[Tuple[int, Tuple[float, float], np.ndarray]] = []

        for frame_idx in unlabeled_frames:
            npy_path = fused_dir / f"frame_{frame_idx:06d}_fused.npy"
            keypoints_3d = load_fused_keypoints(npy_path)
            if keypoints_3d is None:
                continue

            head_kpts = extract_head_keypoints(keypoints_3d, self.keypoint_indices)
            if head_kpts is None:
                continue

            angles = calculate_head_angles(head_kpts)
            candidate_records.append((frame_idx, angles, keypoints_3d.astype(np.float64)))

        if not candidate_records:
            return None

        frame_indices = np.array([item[0] for item in candidate_records], dtype=np.int64)
        pitch_values = np.array([item[1][0] for item in candidate_records], dtype=np.float64)
        yaw_values = np.array([item[1][1] for item in candidate_records], dtype=np.float64)

        robust_ratio = selection_ratio
        robust_min_frames = min_selected_frames
        if max_selected_frames is not None:
            robust_min_frames = min(robust_min_frames, max_selected_frames)

        baseline_pitch, selected_local_idx, candidate_local_idx = estimate_stable_front_baseline(
            raw_pitch_vals=pitch_values,
            yaw_vals=yaw_values,
            front_ratio=robust_ratio,
            min_front_frames=robust_min_frames,
        )

        if candidate_local_idx.size == 0:
            return None

        candidate_frame_count = int(candidate_local_idx.size)
        selected_frame_indices = frame_indices[selected_local_idx] if selected_local_idx.size > 0 else np.array([], dtype=np.int64)

        selected_records = []
        used_frames: List[int] = []
        keypoints_sum = None
        angle_values: List[Tuple[float, float]] = []

        selected_set = set(int(idx) for idx in selected_frame_indices.tolist())
        for frame_idx, angles, keypoints_3d in candidate_records:
            if frame_idx not in selected_set:
                continue
            used_frames.append(frame_idx)
            angle_values.append(angles)
            selected_records.append((frame_idx, angles, keypoints_3d))
            if keypoints_sum is None:
                keypoints_sum = keypoints_3d
            else:
                keypoints_sum += keypoints_3d

        if keypoints_sum is None or not angle_values:
            return None

        mean_keypoints = keypoints_sum / len(selected_records)
        mean_pitch = float(np.mean([angles[0] for angles in angle_values]))
        mean_yaw = float(np.mean([angles[1] for angles in angle_values]))

        return {
            "video_id": video_id,
            "frame_count": len(selected_records),
            "candidate_frame_count": candidate_frame_count,
            "selection_ratio": selection_ratio,
            "frame_indices": used_frames,
            "mean_keypoints_3d": mean_keypoints.tolist(),
            "mean_angles": {
                "pitch": mean_pitch,
                "yaw": mean_yaw,
            },
            "robust_front_baseline_pitch": baseline_pitch,
        }

    def compare_with_annotations(
        self,
        video_id: str,
        frame_idx: int,
        angles: Dict[str, float],
        threshold_deg: float = 15.0,
        baseline_angles: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict]:
        """
        将计算的角度与标注进行比较
        
        Args:
            video_id: 视频ID (例如: "01_day_high")
            frame_idx: 帧索引
            angles: 计算出的角度字典 {'pitch': float, 'yaw': float}
            threshold_deg: 判断是否匹配的阈值（度）
            
        Returns:
            比较结果字典，包含标注信息和匹配结果，如果没有标注则返回None
        """
        if self.annotation_dict is None:
            return None
            
        if video_id not in self.annotation_dict:
            return None
            
        # 获取该帧的所有标注
        frame_annotations = get_all_annotations_for_frame(
            self.annotation_dict[video_id], frame_idx
        )
        
        # 如果没有标注，直接跳过此帧比较
        if not frame_annotations:
            return None

        adjusted_angles = self._apply_baseline(angles, baseline_angles)
        
        # 与每个标注比较
        matches = []
        for annotation in frame_annotations:
            label = annotation.label.lower()
            if label not in LABEL_DIRECTION_MAP:
                continue

            expected_pitch_dir, expected_yaw_dir = LABEL_DIRECTION_MAP[label]

            pitch_value = adjusted_angles.get("pitch", 0)
            yaw_value = adjusted_angles.get("yaw", 0)

            # 检查pitch和yaw是否都匹配
            pitch_match = direction_match(pitch_value, expected_pitch_dir, threshold_deg)
            yaw_match = direction_match(yaw_value, expected_yaw_dir, threshold_deg)
            is_match = pitch_match and yaw_match

            # 分轴语义：只在该标签涉及该轴时参与该轴评估
            pitch_axis_active = expected_pitch_dir != 0
            yaw_axis_active = expected_yaw_dir != 0
            
            matches.append({
                "annotation": annotation,
                "pitch_value": pitch_value,
                "yaw_value": yaw_value,
                "expected_pitch": expected_pitch_dir,
                "expected_yaw": expected_yaw_dir,
                "pitch_axis_active": pitch_axis_active,
                "yaw_axis_active": yaw_axis_active,
                "pitch_match": pitch_match,
                "yaw_match": yaw_match,
                "is_match": is_match,
            })
        
        return {
            "frame_idx": frame_idx,
            "video_id": video_id,
            "angles": angles,
            "adjusted_angles": adjusted_angles,
            "annotations": frame_annotations,
            "matches": matches,
        }

    def analyze_sequence_with_annotations(
        self,
        video_id: str,
        fused_dir: Path,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        threshold_deg: float = 15.0,
        baseline_angles: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        分析序列并与标注进行比较
        
        Args:
            video_id: 视频ID (例如: "01_day_high")
            fused_dir: 融合npy文件目录
            start_frame: 起始帧（可选）
            end_frame: 结束帧（可选）
            threshold_deg: 判断是否匹配的阈值（度）
            
        Returns:
            包含角度和比较结果的字典
        """
        # 分析角度
        angles_results = self.analyze_sequence(fused_dir, start_frame, end_frame)
        
        # 如果没有标注，直接返回角度结果
        if self.annotation_dict is None:
            return {
                "angles": angles_results,
                "comparisons": {},
            }
        
        # 与标注比较
        comparisons = {}
        for frame_idx, angles in angles_results.items():
            comparison = self.compare_with_annotations(
                video_id, frame_idx, angles, threshold_deg=threshold_deg
                , baseline_angles=baseline_angles
            )
            if comparison:
                comparisons[frame_idx] = comparison
        
        return {
            "angles": angles_results,
            "comparisons": comparisons,
            "baseline_angles": baseline_angles,
        }

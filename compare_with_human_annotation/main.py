#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/compare_with_human_annotation/main.py
Project: /workspace/code/compare_with_human_annotation
Created Date: Saturday February 7th 2026
Author: Kaixu Chen

与人工标注进行比较的头部姿态分析器核心模块
只保留与标注比较相关的逻辑，删除可视化、导出、批量处理等功能。
'''
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

    @staticmethod
    def _front_score(angles: Tuple[float, float]) -> float:
        """Smaller score means closer to the front pose."""
        pitch, yaw = angles
        return abs(pitch) + abs(yaw)

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

        candidate_records: List[Tuple[float, int, Tuple[float, float], np.ndarray]] = []

        for frame_idx in unlabeled_frames:
            npy_path = fused_dir / f"frame_{frame_idx:06d}_fused.npy"
            keypoints_3d = load_fused_keypoints(npy_path)
            if keypoints_3d is None:
                continue

            head_kpts = extract_head_keypoints(keypoints_3d, self.keypoint_indices)
            if head_kpts is None:
                continue

            angles = calculate_head_angles(head_kpts)
            score = self._front_score(angles)
            candidate_records.append((score, frame_idx, angles, keypoints_3d.astype(np.float64)))

        if not candidate_records:
            return None

        candidate_records.sort(key=lambda item: item[0])
        selected_count = max(
            min_selected_frames,
            int(len(candidate_records) * selection_ratio),
        )
        selected_count = min(selected_count, max_selected_frames, len(candidate_records))

        selected_records = candidate_records[:selected_count]
        keypoints_sum = None
        angle_values: List[Tuple[float, float, float]] = []
        used_frames: List[int] = []

        for _, frame_idx, angles, keypoints_3d in selected_records:
            angle_values.append(angles)
            used_frames.append(frame_idx)

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
            "candidate_frame_count": len(candidate_records),
            "selection_ratio": selection_ratio,
            "frame_indices": used_frames,
            "mean_keypoints_3d": mean_keypoints.tolist(),
            "mean_angles": {
                "pitch": mean_pitch,
                "yaw": mean_yaw,
            },
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
            
            matches.append({
                "annotation": annotation,
                "pitch_value": pitch_value,
                "yaw_value": yaw_value,
                "expected_pitch": expected_pitch_dir,
                "expected_yaw": expected_yaw_dir,
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


def build_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""
    from .config import DEFAULT_THRESHOLD, SAM3D_VIEWS

    parser = argparse.ArgumentParser(
        prog="python -m compare_with_human_annotation",
        description="头部姿态与人工标注比较工具",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single_parser = subparsers.add_parser(
        "single",
        help="运行 fused 比较；省略 person/env 时遍历全部",
    )
    single_parser.add_argument(
        "person_id",
        nargs="?",
        default=None,
        help="人物 ID；省略 person_id 和 env_jp 时默认遍历全部（fused）",
    )
    single_parser.add_argument(
        "env_jp",
        nargs="?",
        default=None,
        help="环境（日文）；省略 person_id 和 env_jp 时默认遍历全部（fused）",
    )
    single_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="匹配阈值（度）",
    )

    majority_parser = subparsers.add_parser(
        "majority",
        help="先对 3 位标注者做多数投票，再批量运行所有人物与环境",
    )
    majority_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="匹配阈值（度）",
    )
    
    by_annotator_parser = subparsers.add_parser(
        "by_annotator",
        help="分别与每位标注者的标注进行比较，生成 3 份独立结果",
    )
    by_annotator_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="匹配阈值（度）",
    )

    single_view_majority_parser = subparsers.add_parser(
        "single_view_majority",
        help="单视角比较（多数投票标注）；省略 person/env 时遍历全部",
    )
    single_view_majority_parser.add_argument(
        "person_id",
        nargs="?",
        default=None,
        help="人物 ID；省略 person_id 和 env_jp 时默认遍历全部",
    )
    single_view_majority_parser.add_argument(
        "env_jp",
        nargs="?",
        default=None,
        help="环境（日文）；省略 person_id 和 env_jp 时默认遍历全部",
    )
    single_view_majority_parser.add_argument(
        "--view",
        type=str,
        default="all",
        choices=["all", *SAM3D_VIEWS],
        help="指定视角（front/left/right）或 all",
    )
    single_view_majority_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="匹配阈值（度）",
    )

    single_view_by_annotator_parser = subparsers.add_parser(
        "single_view_by_annotator",
        help="单视角比较（按标注者）；省略 person/env 时遍历全部",
    )
    single_view_by_annotator_parser.add_argument(
        "person_id",
        nargs="?",
        default=None,
        help="人物 ID；省略 person_id 和 env_jp 时默认遍历全部",
    )
    single_view_by_annotator_parser.add_argument(
        "env_jp",
        nargs="?",
        default=None,
        help="环境（日文）；省略 person_id 和 env_jp 时默认遍历全部",
    )
    single_view_by_annotator_parser.add_argument(
        "--view",
        type=str,
        default="all",
        choices=["all", *SAM3D_VIEWS],
        help="指定视角（front/left/right）或 all",
    )
    single_view_by_annotator_parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="匹配阈值（度）",
    )
    return parser


def main() -> None:
    """程序入口：需要显式指定运行模式。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "majority":
        from .batch_run import run_batch_comparison_majority_vote

        run_batch_comparison_majority_vote(threshold_deg=args.threshold)
        return

    if args.mode == "by_annotator":
        from .batch_run import run_batch_comparison_by_annotator

        run_batch_comparison_by_annotator(threshold_deg=args.threshold)
        return

    if args.mode == "single":
        from .batch_run import run_batch_comparison
        from .run import run_comparison

        if args.person_id is None and args.env_jp is None:
            run_batch_comparison(threshold_deg=args.threshold)
            return

        run_comparison(
            person_id=args.person_id,
            env_jp=args.env_jp,
            threshold=args.threshold,
        )
        return

    if args.mode == "single_view_majority":
        from .run import run_single_view_comparison, run_single_view_comparison_all

        if args.person_id is None and args.env_jp is None:
            run_single_view_comparison_all(
                view=args.view,
                annotation_mode="majority",
                threshold=args.threshold,
            )
            return

        run_single_view_comparison(
            person_id=args.person_id,
            env_jp=args.env_jp,
            view=args.view,
            annotation_mode="majority",
            threshold=args.threshold,
        )
        return

    if args.mode == "single_view_by_annotator":
        from .run import run_single_view_comparison, run_single_view_comparison_all

        if args.person_id is None and args.env_jp is None:
            run_single_view_comparison_all(
                view=args.view,
                annotation_mode="by_annotator",
                threshold=args.threshold,
            )
            return

        run_single_view_comparison(
            person_id=args.person_id,
            env_jp=args.env_jp,
            view=args.view,
            annotation_mode="by_annotator",
            threshold=args.threshold,
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()

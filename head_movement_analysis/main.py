#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/head_movement_analysis/main.py
Project: /workspace/code/head_movement_analysis
Created Date: Saturday February 7th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday February 7th 2026 12:18:48 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''
import argparse
import csv
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback when tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable

from .load import (
    HeadMovementLabel,
    get_all_annotations_for_frame,
    get_annotation_for_frame,
    load_fused_keypoints,
    load_head_movement_annotations,
    load_multi_annotator_annotations,
)
from .angle_calculator import (
    KEYPOINT_INDICES,
    LABEL_DIRECTION_MAP,
    calculate_head_angles,
    classify_label,
    direction_match,
    extract_head_keypoints,
)

logger = logging.getLogger(__name__)

ENV_MAPPING = {
    "夜多い": "night_high",
    "夜少ない": "night_low",
    "昼多い": "day_high",
    "昼少ない": "day_low",
}

ENV_REVERSE_MAPPING = {value: key for key, value in ENV_MAPPING.items()}


def _to_env_jp(env_name: str) -> str:
    if env_name in ENV_MAPPING:
        return env_name
    return ENV_REVERSE_MAPPING.get(env_name, env_name)


def _resolve_fused_dir_from_args(
    base_dir: Path,
    person_id: str,
    env_name: str,
) -> Path:
    env_jp = _to_env_jp(env_name)
    return Path(base_dir) / person_id / env_jp / "fused_npz"


def _resolve_data_dir_from_args(
    base_dir: Path,
    person_id: str,
    env_name: str,
    data_type: str,
) -> Path:
    env_jp = _to_env_jp(env_name)
    subdir = "smoothed_fused_npz" if data_type == "smoothed" else "fused_npz"
    return Path(base_dir) / person_id / env_jp / subdir


def _expand_data_types(data_type: str) -> List[str]:
    if data_type == "both":
        return ["fused", "smoothed"]
    return [data_type]


def _resolve_single_frame_npy(fused_dir: Path, frame_idx: int) -> Path:
    candidate = fused_dir / f"frame_{frame_idx:06d}_fused.npy"
    if candidate.exists():
        return candidate

    matches = sorted(fused_dir.glob(f"frame_{frame_idx}_fused.npy"))
    if matches:
        return matches[0]

    return candidate


class HeadPoseAnalyzer:
    """
    头部姿态分析器
    从融合后的3D关键点计算头部的三个转动角度：
    1. Pitch（俯仰角）：上下点头
    2. Yaw（偏航角）：左右转头
    3. Roll（翻滚角）：头部左右倾斜
    """

    def __init__(self, annotation_dict: Optional[Dict[str, List[HeadMovementLabel]]] = None):
        """初始化头部姿态分析器
        
        Args:
            annotation_dict: 可选的标注字典，用于与计算结果比较
        """
        self.keypoint_indices = KEYPOINT_INDICES
        self.annotation_dict = annotation_dict

    def analyze_head_pose(self, npy_path: Path) -> Optional[Dict[str, float]]:
        """
        分析单帧的头部姿态

        Args:
            npy_path: 融合后的.npy文件路径

        Returns:
            包含三个角度的字典，如果分析失败则返回None
            {
                'pitch': float,  # 俯仰角（度）
                'yaw': float,    # 偏航角（度）
                'roll': float,   # 翻滚角（度）
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
        pitch, yaw, roll = calculate_head_angles(head_kpts)

        result = {
            "pitch": pitch,
            "yaw": yaw,
            "roll": roll,
        }
        
        return result

    def analyze_sequence(
        self, fused_dir: Path, start_frame: int = None, end_frame: int = None
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
                    'roll': float,
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

    def compare_with_annotations(
        self,
        video_id: str,
        frame_idx: int,
        angles: Dict[str, float],
        threshold_deg: float = 15.0,
    ) -> Optional[Dict]:
        """
        将计算的角度与标注进行比较
        
        Args:
            video_id: 视频ID (例如: "01_day_high")
            frame_idx: 帧索引
            angles: 计算出的角度字典 {'pitch': float, 'yaw': float, 'roll': float}
            threshold_deg: 判断是否匹配的阈值（度）
            
        Returns:
            比较结果字典，包含标注信息和匹配结果
            如果没有标注或没有加载标注字典，返回None
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
        
        # label -> (pitch_dir, yaw_dir)
        matches = []
        for annotation in frame_annotations:
            label = annotation.label.lower()
            if label not in LABEL_DIRECTION_MAP:
                continue

            expected_pitch_dir, expected_yaw_dir = LABEL_DIRECTION_MAP[label]

            pitch_value = angles.get("pitch", 0)
            yaw_value = angles.get("yaw", 0)

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
            "annotations": frame_annotations,
            "matches": matches,
        }

    def analyze_sequence_with_annotations(
        self,
        video_id: str,
        fused_dir: Path,
        start_frame: int = None,
        end_frame: int = None,
    ) -> Dict:
        """
        分析序列并与标注进行比较
        
        Args:
            video_id: 视频ID
            fused_dir: 融合npy文件目录
            start_frame: 起始帧
            end_frame: 结束帧
            
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
            comparison = self.compare_with_annotations(video_id, frame_idx, angles)
            if comparison:
                comparisons[frame_idx] = comparison
        
        return {
            "angles": angles_results,
            "comparisons": comparisons,
        }


class BatchHeadPoseAnalyzer:
    """批量头部姿态分析器"""

    def __init__(self, base_dir: Path, output_dir: Path, data_type: str = "fused"):
        """
        Args:
            base_dir: head3d_fuse_results 基础目录
            output_dir: 输出结果目录
        """
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.data_type = data_type
        self.data_subdir = "smoothed_fused_npz" if data_type == "smoothed" else "fused_npz"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_dir = self.output_dir / "csv"
        self.plot_dir = self.output_dir / "plots"
        self.csv_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

        self.analyzer = HeadPoseAnalyzer()

    def get_all_person_env_pairs(self) -> List[Tuple[str, str, Path]]:
        """
        扫描base_dir获取所有可用的person和env组合

        Returns:
            List of (person_id, env_name, fused_dir_path)
        """
        pairs = []

        for person_dir in sorted(self.base_dir.iterdir()):
            if not person_dir.is_dir():
                continue

            person_id = person_dir.name
            for env_dir in person_dir.iterdir():
                if not env_dir.is_dir():
                    continue

                env_name = env_dir.name
                fused_dir = env_dir / self.data_subdir

                if fused_dir.exists() and any(fused_dir.glob("frame_*_fused.npy")):
                    pairs.append((person_id, env_name, fused_dir))

        return pairs

    def analyze_one_combination(
        self,
        person_id: str,
        env_name: str,
        fused_dir: Path,
    ) -> Dict[int, Dict[str, float]]:
        """分析一个person和env组合"""
        logger.info(f"分析 {person_id}/{env_name}")

        try:
            results = self.analyzer.analyze_sequence(fused_dir)
            logger.info(f"  成功分析 {len(results)} 帧")
            return results
        except Exception as exc:
            logger.error(f"  分析失败: {exc}")
            return {}

    def save_results_to_csv(
        self,
        person_id: str,
        env_name: str,
        results: Dict[int, Dict[str, float]],
    ) -> Path:
        """保存结果到CSV文件"""
        env_en = ENV_MAPPING.get(env_name, env_name)
        csv_path = self.csv_dir / f"person_{person_id}_{env_en}.csv"

        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["frame_idx", "pitch_deg", "yaw_deg", "roll_deg"])

            for frame_idx in sorted(results.keys()):
                angles = results[frame_idx]
                writer.writerow(
                    [
                        frame_idx,
                        f"{angles['pitch']:.2f}",
                        f"{angles['yaw']:.2f}",
                        f"{angles['roll']:.2f}",
                    ]
                )

        logger.debug(f"  保存CSV: {csv_path}")
        return csv_path

    def create_visualization(
        self,
        person_id: str,
        env_name: str,
        results: Dict[int, Dict[str, float]],
    ) -> Optional[Path]:
        """创建角度变化可视化图表"""
        if not results:
            return None

        frame_indices = sorted(results.keys())
        pitches = [results[idx]["pitch"] for idx in frame_indices]
        yaws = [results[idx]["yaw"] for idx in frame_indices]
        rolls = [results[idx]["roll"] for idx in frame_indices]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(
            f"Head Pose Angles - Person {person_id} - {env_name}",
            fontsize=16,
            fontweight="bold",
        )

        axes[0].plot(frame_indices, pitches, "b-", linewidth=1.5, alpha=0.8)
        axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_ylabel("Pitch (degrees)", fontsize=11)
        axes[0].set_title("Pitch (Up/Down)", fontsize=10)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(frame_indices, yaws, "g-", linewidth=1.5, alpha=0.8)
        axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_ylabel("Yaw (degrees)", fontsize=11)
        axes[1].set_title("Yaw (Left/Right)", fontsize=10)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(frame_indices, rolls, "r-", linewidth=1.5, alpha=0.8)
        axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        axes[2].set_xlabel("Frame Index", fontsize=11)
        axes[2].set_ylabel("Roll (degrees)", fontsize=11)
        axes[2].set_title("Roll (Tilt)", fontsize=10)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        env_en = ENV_MAPPING.get(env_name, env_name)
        plot_path = self.plot_dir / f"person_{person_id}_{env_en}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.debug(f"  保存图表: {plot_path}")
        return plot_path

    def run_batch_analysis(
        self,
        person_filter: Optional[List[str]] = None,
        env_filter: Optional[List[str]] = None,
        save_csv: bool = True,
        save_plot: bool = True,
    ) -> Dict:
        """运行批量分析"""
        logger.info("=" * 60)
        logger.info("开始批量头部姿态分析")
        logger.info("=" * 60)

        all_pairs = self.get_all_person_env_pairs()
        logger.info(f"找到 {len(all_pairs)} 个person-env组合")

        if person_filter:
            all_pairs = [(p, e, d) for p, e, d in all_pairs if p in person_filter]
            logger.info(f"应用person过滤后: {len(all_pairs)} 个组合")

        if env_filter:
            all_pairs = [(p, e, d) for p, e, d in all_pairs if e in env_filter]
            logger.info(f"应用env过滤后: {len(all_pairs)} 个组合")

        stats = {
            "total_combinations": len(all_pairs),
            "successful": 0,
            "failed": 0,
            "total_frames": 0,
            "results": {},
        }

        logger.info("\n开始处理...")
        for person_id, env_name, fused_dir in tqdm(all_pairs, desc="处理进度"):
            try:
                results = self.analyze_one_combination(person_id, env_name, fused_dir)

                if results:
                    stats["successful"] += 1
                    stats["total_frames"] += len(results)

                    csv_path = None
                    plot_path = None
                    if save_csv:
                        csv_path = self.save_results_to_csv(person_id, env_name, results)
                    if save_plot:
                        plot_path = self.create_visualization(person_id, env_name, results)

                    key = f"{person_id}_{env_name}"
                    stats["results"][key] = {
                        "frames": len(results),
                        "csv": str(csv_path) if csv_path else None,
                        "plot": str(plot_path) if plot_path else None,
                    }
                else:
                    stats["failed"] += 1
            except Exception as exc:
                logger.error(f"处理 {person_id}/{env_name} 时出错: {exc}")
                stats["failed"] += 1

        logger.info("\n" + "=" * 60)
        logger.info("批量分析完成!")
        logger.info("=" * 60)
        logger.info(f"总组合数: {stats['total_combinations']}")
        logger.info(f"成功: {stats['successful']}")
        logger.info(f"失败: {stats['failed']}")
        logger.info(f"总帧数: {stats['total_frames']}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"  - CSV: {self.csv_dir}")
        logger.info(f"  - 图表: {self.plot_dir}")

        stats_file = self.output_dir / "batch_stats.txt"
        with stats_file.open("w") as handle:
            handle.write("批量头部姿态分析统计\n")
            handle.write("=" * 60 + "\n\n")
            handle.write(f"总组合数: {stats['total_combinations']}\n")
            handle.write(f"成功: {stats['successful']}\n")
            handle.write(f"失败: {stats['failed']}\n")
            handle.write(f"总帧数: {stats['total_frames']}\n\n")
            handle.write("详细结果:\n")
            for key, info in sorted(stats["results"].items()):
                handle.write(f"\n{key}:\n")
                handle.write(f"  帧数: {info['frames']}\n")
                if info["csv"]:
                    handle.write(f"  CSV: {info['csv']}\n")
                if info["plot"]:
                    handle.write(f"  图表: {info['plot']}\n")

        logger.info(f"\n统计信息已保存到: {stats_file}")
        return stats


def _normalize_env_list(envs: Optional[List[str]]) -> List[str]:
    if not envs:
        return ["day_high", "day_low", "night_high", "night_low"]

    normalized = []
    for env in envs:
        if env in ENV_REVERSE_MAPPING:
            normalized.append(env)
        elif env in ENV_MAPPING:
            normalized.append(ENV_MAPPING[env])
        else:
            normalized.append(env)
    return normalized


def _prepare_output_dirs(output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subdirs = {
        "angles": output_dir / "angles_csv",
        "comparisons": output_dir / "comparisons_csv",
        "plots": output_dir / "plots",
        "stats": output_dir / "stats",
    }
    for path in subdirs.values():
        path.mkdir(exist_ok=True)
    return subdirs


def _prepare_output_dirs_for_pair(
    output_dir: Path,
    person_id: str,
    env_en: str,
) -> Dict[str, Path]:
    pair_dir = output_dir / f"person_{person_id}" / env_en
    return _prepare_output_dirs(pair_dir)


def _plot_angles_with_annotations(
    frame_indices: List[int],
    angles: Dict[int, Dict[str, float]],
    labels: List[HeadMovementLabel],
    output_path: Path,
    title: str,
) -> None:
    pitches = [angles[idx]["pitch"] for idx in frame_indices]
    yaws = [angles[idx]["yaw"] for idx in frame_indices]
    rolls = [angles[idx]["roll"] for idx in frame_indices]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    axes[0].plot(frame_indices, pitches, "b-", linewidth=1.5, alpha=0.8)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Pitch (degrees)", fontsize=11)
    axes[0].set_title("Pitch (Up/Down)", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frame_indices, yaws, "g-", linewidth=1.5, alpha=0.8)
    axes[1].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_ylabel("Yaw (degrees)", fontsize=11)
    axes[1].set_title("Yaw (Left/Right)", fontsize=10)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frame_indices, rolls, "r-", linewidth=1.5, alpha=0.8)
    axes[2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xlabel("Frame Index", fontsize=11)
    axes[2].set_ylabel("Roll (degrees)", fontsize=11)
    axes[2].set_title("Roll (Tilt)", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    label_colors = {
        "up": "#2ca02c",
        "down": "#d62728",
        "left": "#1f77b4",
        "right": "#ff7f0e",
        "front": "#7f7f7f",
    }

    for label in labels:
        label_name = label.label.lower()
        if label_name == "front":
            continue
        color = label_colors.get(label_name, "#7f7f7f")
        start = label.start_frame
        end = label.end_frame

        if label_name in {"up", "down"}:
            axes[0].axvspan(start, end, color=color, alpha=0.15)
        elif label_name in {"left", "right"}:
            axes[1].axvspan(start, end, color=color, alpha=0.15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_match_rates(
    match_stats: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str,
) -> None:
    annotators = list(match_stats.keys())
    overall_rates = [match_stats[name]["overall_rate"] for name in annotators]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(annotators, overall_rates, color="#4c72b0", alpha=0.8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Match Rate (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)

    for idx, rate in enumerate(overall_rates):
        ax.text(idx, rate + 1, f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _process_compare_pair(
    person_id: str,
    env_en: str,
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    env_jp = ENV_REVERSE_MAPPING.get(env_en, env_en)
    data_subdir = getattr(args, "data_subdir", "fused_npz")
    data_type = getattr(args, "active_data_type", "fused")
    fused_dir = args.base_dir / person_id / env_jp / data_subdir
    if not fused_dir.exists():
        logger.warning("Missing fused dir: %s", fused_dir)
        return []

    label_file = args.label_dir / f"person_{person_id}_{env_en}_h265.json"
    video_id, labels_by_annotator = load_multi_annotator_annotations(label_file)
    if not video_id or not labels_by_annotator:
        logger.warning("No labels found: %s", label_file)
        return []

    logger.info("Processing %s %s", person_id, env_en)

    analyzer = HeadPoseAnalyzer()
    angles = analyzer.analyze_sequence(fused_dir)
    if not angles:
        logger.warning("No angles computed for %s", fused_dir)
        return []

    output_dirs = _prepare_output_dirs_for_pair(args.output_dir, person_id, env_en)
    angles_dir = output_dirs["angles"]
    comparisons_dir = output_dirs["comparisons"]
    plots_dir = output_dirs["plots"]

    frame_indices = sorted(angles.keys())
    angles_csv = angles_dir / "angles.csv"
    with angles_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "pitch_deg", "yaw_deg", "roll_deg"])
        for frame_idx in frame_indices:
            angle = angles[frame_idx]
            writer.writerow(
                [
                    frame_idx,
                    f"{angle['pitch']:.2f}",
                    f"{angle['yaw']:.2f}",
                    f"{angle['roll']:.2f}",
                ]
            )

    match_stats: Dict[str, Dict[str, float]] = {}
    summary_rows: List[Dict[str, object]] = []

    for idx, labels in enumerate(labels_by_annotator, start=1):
        annotator_name = f"annotator_{idx}"
        annotation_dict = {video_id: labels}
        annotator_analyzer = HeadPoseAnalyzer(annotation_dict=annotation_dict)

        comparisons = {}
        for frame_idx, angle in angles.items():
            comparison = annotator_analyzer.compare_with_annotations(
                video_id,
                frame_idx,
                angle,
                threshold_deg=args.threshold,
            )
            if comparison:
                comparisons[frame_idx] = comparison

        match_stats[annotator_name] = _compute_match_stats(comparisons)

        comparison_csv = comparisons_dir / f"{annotator_name}_compare.csv"
        _write_comparison_csv(
            comparison_csv,
            frame_indices,
            angles,
            labels,
            comparisons,
            args.threshold,
        )

        plot_path = plots_dir / f"{annotator_name}_angles.png"
        _plot_angles_with_annotations(
            frame_indices,
            angles,
            labels,
            plot_path,
            f"Person {person_id} - {env_en} - {annotator_name}",
        )

        summary_rows.append(
            {
                "person": person_id,
                "env": env_en,
                "data_type": data_type,
                "annotator": annotator_name,
                "overall_rate": match_stats[annotator_name]["overall_rate"],
                "total_labels": match_stats[annotator_name]["total_labels"],
                "matched_labels": match_stats[annotator_name]["matched_labels"],
            }
        )

    match_plot_path = plots_dir / "match_rates.png"
    _plot_match_rates(
        match_stats,
        match_plot_path,
        f"Match Rate - Person {person_id} - {env_en}",
    )

    return summary_rows


def _write_comparison_csv(
    output_path: Path,
    frame_indices: List[int],
    angles: Dict[int, Dict[str, float]],
    labels: List[HeadMovementLabel],
    comparisons: Dict[int, Dict],
    threshold: float,
) -> None:
    labels_by_frame = defaultdict(list)
    for label in labels:
        for frame_idx in range(label.start_frame, label.end_frame + 1):
            labels_by_frame[frame_idx].append(label.label)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_idx",
                "pitch_deg",
                "yaw_deg",
                "roll_deg",
                "predicted_label",
                "label_match",
                "labels",
                "match_count",
                "total_labels",
                "match_rate",
            ]
        )

        for frame_idx in frame_indices:
            angle = angles[frame_idx]
            predicted_label = classify_label(
                angle["pitch"],
                angle["yaw"],
                threshold,
            )
            frame_labels = labels_by_frame.get(frame_idx, [])
            if not frame_labels:
                frame_labels = ["front"]
            label_match = int(predicted_label in frame_labels)
            match_count = 0
            total_labels = 0
            match_rate = 0.0
            comparison = comparisons.get(frame_idx)
            if comparison:
                total_labels = len(comparison["matches"])
                match_count = sum(1 for item in comparison["matches"] if item["is_match"])
                if total_labels > 0:
                    match_rate = match_count / total_labels

            writer.writerow(
                [
                    frame_idx,
                    f"{angle['pitch']:.2f}",
                    f"{angle['yaw']:.2f}",
                    f"{angle['roll']:.2f}",
                    predicted_label,
                    label_match,
                    ";".join(sorted(set(frame_labels))),
                    match_count,
                    total_labels,
                    f"{match_rate:.3f}",
                ]
            )


def _compute_match_stats(comparisons: Dict[int, Dict]) -> Dict[str, float]:
    label_totals = defaultdict(int)
    label_matches = defaultdict(int)
    total = 0
    matched = 0

    for comparison in comparisons.values():
        for item in comparison["matches"]:
            label = item["annotation"].label.lower()
            label_totals[label] += 1
            total += 1
            if item["is_match"]:
                label_matches[label] += 1
                matched += 1

    overall_rate = (matched / total * 100) if total > 0 else 0.0
    stats = {
        "overall_rate": overall_rate,
        "total_labels": total,
        "matched_labels": matched,
    }

    for label, total_count in label_totals.items():
        match_count = label_matches.get(label, 0)
        rate = (match_count / total_count * 100) if total_count > 0 else 0.0
        stats[f"{label}_rate"] = rate
        stats[f"{label}_total"] = total_count
        stats[f"{label}_matched"] = match_count

    return stats


def _add_logging(log_level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="头部姿态分析工具（单帧/序列/批量）",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    single_parser = subparsers.add_parser("single", help="分析单帧npy文件")
    single_parser.add_argument("--npy", type=Path, default=None, help="单帧npy文件路径")
    single_parser.add_argument(
        "--base-dir",
        type=Path,
        default="/workspace/data/head3d_fuse_results",
        help="head3d_fuse_results 基础目录（当不传 --npy 时使用）",
    )
    single_parser.add_argument("--person", type=str, default=None, help="person ID，例如 01")
    single_parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="环境名，支持 日文(夜多い/夜少ない/昼多い/昼少ない) 或 英文(day_high/day_low/night_high/night_low)",
    )
    single_parser.add_argument("--frame", type=int, default=None, help="帧索引，例如 619")
    single_parser.add_argument(
        "--data-type",
        type=str,
        default="both",
        choices=["fused", "smoothed", "both"],
        help="输入数据类型：fused、smoothed 或 both",
    )
    single_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    sequence_parser = subparsers.add_parser("sequence", help="分析序列目录")
    sequence_parser.add_argument("--fused-dir", type=Path, default=None, help="融合npy目录")
    sequence_parser.add_argument(
        "--base-dir",
        type=Path,
        default="/workspace/data/head3d_fuse_results",
        help="head3d_fuse_results 基础目录（当不传 --fused-dir 时使用）",
    )
    sequence_parser.add_argument("--person", type=str, default=None, help="person ID，例如 01")
    sequence_parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="环境名，支持 日文(夜多い/夜少ない/昼多い/昼少ない) 或 英文(day_high/day_low/night_high/night_low)",
    )
    sequence_parser.add_argument("--start-frame", type=int, default=None)
    sequence_parser.add_argument("--end-frame", type=int, default=None)
    sequence_parser.add_argument(
        "--data-type",
        type=str,
        default="both",
        choices=["fused", "smoothed", "both"],
        help="输入数据类型：fused、smoothed 或 both",
    )
    sequence_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    batch_parser = subparsers.add_parser("batch", help="批量分析所有person和env")
    batch_parser.add_argument(
        "--base-dir",
        "-b",
        type=Path,
        default="/workspace/data/head3d_fuse_results",
        help="head3d_fuse_results 基础目录",
    )
    batch_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default="/workspace/data/head_pose_analysis_results",
        help="输出结果目录",
    )
    batch_parser.add_argument(
        "--persons",
        "-p",
        nargs="+",
        help="只处理指定的person IDs (例如: 01 02 03)",
    )
    batch_parser.add_argument(
        "--envs",
        "-e",
        nargs="+",
        help="只处理指定的环境 (例如: 夜多い 昼多い)",
    )
    batch_parser.add_argument("--no-csv", action="store_true", help="不保存CSV文件")
    batch_parser.add_argument("--no-plot", action="store_true", help="不生成图表")
    batch_parser.add_argument(
        "--data-type",
        type=str,
        default="both",
        choices=["fused", "smoothed", "both"],
        help="输入数据类型：fused、smoothed 或 both",
    )
    batch_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    compare_parser = subparsers.add_parser("compare", help="与多标注者标注进行比较")
    compare_parser.add_argument(
        "--label-dir",
        type=Path,
        default="/workspace/data/multi_view_driver_action/label",
        help="标注文件目录",
    )
    compare_parser.add_argument(
        "--base-dir",
        type=Path,
        default="/workspace/data/head3d_fuse_results",
        help="head3d_fuse_results 基础目录",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default="/workspace/code/logs",
        help="输出结果目录",
    )
    compare_parser.add_argument(
        "--persons",
        "-p",
        nargs="+",
        help="要比较的person IDs (例如: 03 04 05)",
    )
    compare_parser.add_argument(
        "--envs",
        "-e",
        nargs="+",
        help="只处理指定的环境 (例如: day_high 夜多い)",
    )
    compare_parser.add_argument(
        "--threshold",
        type=float,
        default=15.0,
        help="匹配阈值（度）",
    )
    compare_parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="并行线程数（按person/env处理）",
    )
    compare_parser.add_argument(
        "--data-type",
        type=str,
        default="both",
        choices=["fused", "smoothed", "both"],
        help="输入数据类型：fused、smoothed 或 both",
    )
    compare_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别",
    )

    args = parser.parse_args()
    _add_logging(args.log_level)

    if args.command == "single":
        analyzer = HeadPoseAnalyzer()

        data_types = _expand_data_types(args.data_type)
        if args.npy is not None and len(data_types) > 1:
            logger.warning(
                "single 模式传入 --npy 时，--data-type=both 无法自动拆分，按单路径执行"
            )
            data_types = ["fused"]

        for data_type in data_types:
            npy_path = args.npy
            if npy_path is None:
                if args.person is None or args.env is None or args.frame is None:
                    logger.error(
                        "single 模式在不传 --npy 时，需要提供 --person --env --frame"
                    )
                    return

                fused_dir = _resolve_data_dir_from_args(
                    args.base_dir,
                    args.person,
                    args.env,
                    data_type,
                )
                if not fused_dir.exists():
                    logger.error("Data directory does not exist (%s): %s", data_type, fused_dir)
                    continue

                npy_path = _resolve_single_frame_npy(fused_dir, args.frame)
                if not npy_path.exists():
                    logger.error("Frame file not found (%s): %s", data_type, npy_path)
                    continue

                logger.info("Resolved input npy (%s): %s", data_type, npy_path)

            result = analyzer.analyze_head_pose(npy_path)
            if result:
                logger.info("单帧分析结果 (%s):", data_type)
                logger.info(f"  俯仰角 (Pitch): {result['pitch']:.2f}°")
                logger.info(f"  偏航角 (Yaw): {result['yaw']:.2f}°")
                logger.info(f"  翻滚角 (Roll): {result['roll']:.2f}°")
        return

    if args.command == "sequence":
        analyzer = HeadPoseAnalyzer()

        data_types = _expand_data_types(args.data_type)
        if args.fused_dir is not None and len(data_types) > 1:
            logger.warning(
                "sequence 模式传入 --fused-dir 时，--data-type=both 无法自动拆分，按单目录执行"
            )
            data_types = ["fused"]

        for data_type in data_types:
            fused_dir = args.fused_dir
            if fused_dir is None:
                if args.person is None or args.env is None:
                    logger.error(
                        "sequence 模式在不传 --fused-dir 时，需要提供 --person --env"
                    )
                    return
                fused_dir = _resolve_data_dir_from_args(
                    args.base_dir,
                    args.person,
                    args.env,
                    data_type,
                )

            if not fused_dir.exists():
                logger.error("Data directory does not exist (%s): %s", data_type, fused_dir)
                continue

            logger.info("Using %s directory: %s", data_type, fused_dir)
            results = analyzer.analyze_sequence(
                fused_dir,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
            )
            logger.info("\n序列分析结果 (%s, 前10帧):", data_type)
            for frame_idx in sorted(results.keys())[:10]:
                angles = results[frame_idx]
                logger.info(
                    f"  Frame {frame_idx}: "
                    f"Pitch={angles['pitch']:6.2f}°, "
                    f"Yaw={angles['yaw']:6.2f}°, "
                    f"Roll={angles['roll']:6.2f}°"
                )
        return

    if args.command == "compare":
        if not args.label_dir.exists():
            logger.error(f"Label directory does not exist: {args.label_dir}")
            return
        if not args.base_dir.exists():
            logger.error(f"Base directory does not exist: {args.base_dir}")
            return

        env_list = _normalize_env_list(args.envs)

        if args.persons:
            person_ids = args.persons
        else:
            person_ids = []
            for json_path in args.label_dir.glob("person_*_h265.json"):
                parts = json_path.name.split("_")
                if len(parts) >= 2:
                    person_ids.append(parts[1])
            person_ids = sorted(set(person_ids))

        if not person_ids:
            logger.error("No person IDs found in %s", args.label_dir)
            return

        tasks = [(person_id, env_en) for person_id in person_ids for env_en in env_list]
        if not tasks:
            logger.error("No person/env pairs to process")
            return

        for data_type in _expand_data_types(args.data_type):
            compare_output_dir = args.output_dir / data_type
            summary_output_dirs = _prepare_output_dirs(compare_output_dir / "summary")
            stats_dir = summary_output_dirs["stats"]
            summary_rows = []

            args.output_dir = compare_output_dir
            args.data_subdir = "smoothed_fused_npz" if data_type == "smoothed" else "fused_npz"
            args.active_data_type = data_type

            logger.info("Start compare for data_type=%s", data_type)

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                future_map = {
                    executor.submit(_process_compare_pair, person_id, env_en, args): (person_id, env_en)
                    for person_id, env_en in tasks
                }
                for future in as_completed(future_map):
                    person_id, env_en = future_map[future]
                    try:
                        rows = future.result()
                        summary_rows.extend(rows)
                    except Exception as exc:
                        logger.error("Failed %s %s (%s): %s", person_id, env_en, data_type, exc)

            if summary_rows:
                summary_csv = stats_dir / "comparison_summary.csv"
                with summary_csv.open("w", newline="") as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=[
                            "person",
                            "env",
                            "data_type",
                            "annotator",
                            "overall_rate",
                            "total_labels",
                            "matched_labels",
                        ],
                    )
                    writer.writeheader()
                    writer.writerows(summary_rows)

                logger.info("Summary saved to %s", summary_csv)

            logger.info("Comparison outputs saved under %s", compare_output_dir)
        return

    if not args.base_dir.exists():
        logger.error(f"Base directory does not exist: {args.base_dir}")
        return

    for data_type in _expand_data_types(args.data_type):
        batch_output_dir = args.output_dir / data_type
        batch_analyzer = BatchHeadPoseAnalyzer(
            base_dir=args.base_dir,
            output_dir=batch_output_dir,
            data_type=data_type,
        )
        batch_analyzer.run_batch_analysis(
            person_filter=args.persons,
            env_filter=args.envs,
            save_csv=not args.no_csv,
            save_plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()

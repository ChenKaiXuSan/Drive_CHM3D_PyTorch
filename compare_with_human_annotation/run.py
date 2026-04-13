#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""与人工标注比较的运行脚本（Hydra 配置版）。"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import DictConfig

from .head_pose_analyzer import HeadPoseAnalyzer
from .load import (
    get_unlabeled_frame_indices,
    load_annotations_by_annotator,
    load_head_movement_annotations,
    load_majority_voted_annotations,
    load_sam3d_keypoints,
)
from .angle_calculator import (
    calculate_head_angles,
    estimate_stable_front_baseline,
    extract_head_keypoints,
)
from .hydra_utils import (
    get_annotation_file,
    get_env_mapping,
    get_fused_dir,
    get_output_root,
    get_output_dir,
    get_single_view_output_root,
    get_sam3d_output_dir,
    get_sam3d_root,
    get_sam3d_view_dir,
    get_sam3d_views,
    load_config,
)


def _get_all_sam3d_person_env_pairs(
    selected_views: Sequence[str],
    cfg: DictConfig,
) -> List[Tuple[str, str]]:
    """获取包含单视角 SAM3D 结果的全部人物-环境组合。"""
    pairs: List[Tuple[str, str]] = []

    sam3d_root = get_sam3d_root(cfg)
    if not sam3d_root.exists():
        return pairs

    for person_dir in sorted(sam3d_root.iterdir()):
        if not person_dir.is_dir():
            continue

        for env_dir in sorted(person_dir.iterdir()):
            if not env_dir.is_dir():
                continue

            has_data = False
            for current_view in selected_views:
                view_dir = env_dir / current_view
                if view_dir.exists() and any(view_dir.glob("*_sam3d_body.npz")):
                    has_data = True
                    break

            if has_data:
                pairs.append((person_dir.name, env_dir.name))

    return pairs


def run_comparison(
    person_id: Optional[str] = None,
    env_jp: Optional[str] = None,
    annotation_mode: str = "majority",
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    threshold: Optional[float] = None,
    cfg: Optional[DictConfig] = None,
):
    """运行 fused 结果与人工标注比较。"""
    cfg = cfg or load_config()
    env_mapping = get_env_mapping(cfg)

    person_id = person_id or str(cfg.defaults_run.person_id)
    env_jp = env_jp or str(cfg.defaults_run.env_jp)
    if start_frame is None:
        start_frame = cfg.defaults_run.start_frame
    if end_frame is None:
        end_frame = cfg.defaults_run.end_frame
    threshold = float(threshold if threshold is not None else cfg.defaults_run.threshold)

    print("=" * 60)
    print("头部姿态与人工标注比较")
    print("=" * 60)
    print("\n配置:")
    print(f"  Person ID: {person_id}")
    print(f"  Environment: {env_jp}")
    print(f"  Frame范围: {start_frame} ~ {end_frame}")
    print(f"  匹配阈值: {threshold}°")

    annotation_file = get_annotation_file(cfg)
    fused_dir = get_fused_dir(cfg, person_id, env_jp)
    env_en = env_mapping.get(env_jp, env_jp)
    output_dir = get_output_dir(
        person_id,
        env_en,
        output_root=get_output_root(cfg, threshold),
    )

    print("\n路径:")
    print(f"  标注文件: {annotation_file}")
    print(f"  融合数据: {fused_dir}")
    print(f"  输出目录: {output_dir}")

    if not annotation_file.exists():
        print(f"\n✗ 错误: 标注文件不存在 {annotation_file}")
        return

    if not fused_dir.exists():
        print(f"\n✗ 错误: 融合数据目录不存在 {fused_dir}")
        return

    annotation_mode = annotation_mode.lower().strip()
    if annotation_mode not in {"majority", "by_annotator"}:
        raise ValueError(f"Unsupported annotation_mode: {annotation_mode}")

    video_id = f"{person_id}_{env_en}"

    print("\n加载标注...")
    annotation_jobs: List[Tuple[str, Dict]] = []
    if annotation_mode == "majority":
        annotations = load_majority_voted_annotations(annotation_file)
        annotation_jobs.append(("majority", annotations))
    else:
        by_annotator = load_annotations_by_annotator(annotation_file)
        max_annotators = max((len(items) for items in by_annotator.values()), default=0)
        for annotator_idx in range(max_annotators):
            annotator_annotations: Dict[str, List] = {}
            for current_video_id, labels_by_annotator in by_annotator.items():
                if annotator_idx >= len(labels_by_annotator):
                    continue
                if labels_by_annotator[annotator_idx]:
                    annotator_annotations[current_video_id] = labels_by_annotator[annotator_idx]
            if annotator_annotations:
                annotation_jobs.append((f"annotator_{annotator_idx + 1}", annotator_annotations))

    print(f"✓ 已加载 {len(annotation_jobs)} 组标注配置")

    for annotation_name, annotations in annotation_jobs:
        if video_id not in annotations:
            print(f"\n✗ 跳过 {annotation_name}: 未找到视频 {video_id} 的标注")
            continue

        print(f"\n---- 标注来源: {annotation_name} ----")
        print("\n创建分析器...")
        analyzer = HeadPoseAnalyzer(annotation_dict=annotations)
        print("✓ 分析器已创建")

        print("\n估计 front baseline...")
        front_baseline = analyzer.estimate_front_baseline(
            video_id=video_id,
            fused_dir=fused_dir,
            start_frame=start_frame,
            end_frame=end_frame,
        )
        if front_baseline is None:
            print("✗ 没有可用于估计 front baseline 的无标注帧")
        else:
            mean_angles = front_baseline["mean_angles"]
            print(
                f"✓ 已从 {front_baseline['candidate_frame_count']} 个无标注帧中选取 "
                f"{front_baseline['frame_count']} 个 front-like 帧估计 front baseline"
            )
            print(
                f"  平均 front angles: Pitch={mean_angles['pitch']:.2f}°, "
                f"Yaw={mean_angles['yaw']:.2f}°"
            )

        print("\n分析并与标注比较...")
        results = analyzer.analyze_sequence_with_annotations(
            video_id=video_id,
            fused_dir=fused_dir,
            start_frame=start_frame,
            end_frame=end_frame,
            threshold_deg=threshold,
            baseline_angles=front_baseline["mean_angles"] if front_baseline else None,
        )

        results["front_baseline"] = front_baseline

        angles = results["angles"]
        comparisons = results["comparisons"]

        print("✓ 分析完成")
        print(f"  分析帧数: {len(angles)}")
        print(f"  有标注的帧: {len(comparisons)}")

        if not comparisons:
            print("\n✗ 没有找到有标注的帧")
            continue

        total_annotations = sum(len(comp["matches"]) for comp in comparisons.values())

        yaw_eval_count = sum(
            sum(1 for m in comp["matches"] if m.get("yaw_axis_active", False))
            for comp in comparisons.values()
        )
        yaw_match_count = sum(
            sum(
                1
                for m in comp["matches"]
                if m.get("yaw_axis_active", False) and m.get("yaw_match", False)
            )
            for comp in comparisons.values()
        )
        yaw_match_rate = (
            yaw_match_count / yaw_eval_count * 100 if yaw_eval_count > 0 else 0.0
        )

        pitch_eval_count = sum(
            sum(1 for m in comp["matches"] if m.get("pitch_axis_active", False))
            for comp in comparisons.values()
        )
        pitch_match_count = sum(
            sum(
                1
                for m in comp["matches"]
                if m.get("pitch_axis_active", False)
                and m.get("pitch_match", False)
            )
            for comp in comparisons.values()
        )
        pitch_match_rate = (
            pitch_match_count / pitch_eval_count * 100 if pitch_eval_count > 0 else 0.0
        )

        print("\n分轴匹配统计:")
        print(f"  总标注数: {total_annotations}")
        print(
            f"  左右(Yaw)分轴匹配率: {yaw_match_rate:.1f}% "
            f"({yaw_match_count}/{yaw_eval_count})"
        )
        print(
            f"  上下(Pitch)分轴匹配率: {pitch_match_rate:.1f}% "
            f"({pitch_match_count}/{pitch_eval_count})"
        )

        print("\n前5帧的详细结果:")
        for frame_idx, comparison in sorted(list(comparisons.items())[:5]):
            print(f"\n  Frame {frame_idx}:")
            for match in comparison["matches"]:
                axis_items = []
                if match.get("pitch_axis_active", False):
                    axis_items.append(
                        f"Pitch={'✓' if match.get('pitch_match', False) else '✗'}"
                    )
                if match.get("yaw_axis_active", False):
                    axis_items.append(
                        f"Yaw={'✓' if match.get('yaw_match', False) else '✗'}"
                    )
                axis_text = ", ".join(axis_items) if axis_items else "No axis active"
                print(
                    f"    {match['annotation'].label}: "
                    f"Pitch={match['pitch_value']:6.1f}°, "
                    f"Yaw={match['yaw_value']:6.1f}° | {axis_text}"
                )
        print("\n" + "-" * 40)

    print("\n" + "=" * 60)


def _parse_sam3d_frame_idx(npz_name: str) -> Optional[int]:
    """解析 *_sam3d_body.npz 文件名中的帧号。"""
    stem = npz_name.replace("_sam3d_body.npz", "")
    if stem.isdigit():
        return int(stem)
    return None


def _estimate_front_baseline_from_angles(
    analyzer: HeadPoseAnalyzer,
    video_id: str,
    angles_by_frame: Dict[int, Dict[str, float]],
    selection_ratio: float = 0.15,
    min_selected_frames: int = 30,
    max_selected_frames: int = 500,
) -> Optional[Dict]:
    """使用未标注帧中最 front-like 的子集估计 baseline。"""
    if analyzer.annotation_dict is None or video_id not in analyzer.annotation_dict:
        return None

    frame_indices = sorted(angles_by_frame.keys())
    unlabeled_frames = get_unlabeled_frame_indices(
        analyzer.annotation_dict[video_id],
        frame_indices,
    )
    if not unlabeled_frames:
        return None

    candidate_frames: List[int] = []
    raw_pitch_vals: List[float] = []
    yaw_vals: List[float] = []

    for frame_idx in unlabeled_frames:
        angles = angles_by_frame.get(frame_idx)
        if angles is None:
            continue
        candidate_frames.append(frame_idx)
        raw_pitch_vals.append(float(angles["pitch"]))
        yaw_vals.append(float(angles["yaw"]))

    if not candidate_frames:
        return None

    raw_pitch_arr = np.asarray(raw_pitch_vals, dtype=np.float64)
    yaw_arr = np.asarray(yaw_vals, dtype=np.float64)

    robust_min_frames = min_selected_frames
    if max_selected_frames is not None:
        robust_min_frames = min(robust_min_frames, max_selected_frames)

    baseline_pitch, selected_local_idx, candidate_local_idx = estimate_stable_front_baseline(
        raw_pitch_vals=raw_pitch_arr,
        yaw_vals=yaw_arr,
        front_ratio=selection_ratio,
        min_front_frames=robust_min_frames,
    )

    if candidate_local_idx.size == 0:
        return None

    selected_frame_indices = np.asarray(candidate_frames, dtype=np.int64)[selected_local_idx]
    selected_pitch = raw_pitch_arr[selected_local_idx]
    selected_yaw = yaw_arr[selected_local_idx]

    mean_pitch = float(np.mean(selected_pitch)) if selected_pitch.size > 0 else 0.0
    mean_yaw = float(np.mean(selected_yaw)) if selected_yaw.size > 0 else 0.0

    return {
        "video_id": video_id,
        "frame_count": int(selected_frame_indices.size),
        "candidate_frame_count": int(candidate_local_idx.size),
        "selection_ratio": selection_ratio,
        "frame_indices": selected_frame_indices.tolist(),
        "mean_angles": {
            "pitch": mean_pitch,
            "yaw": mean_yaw,
        },
        "robust_front_baseline_pitch": baseline_pitch,
    }


def _serialize_comparison(comparison: Dict) -> Dict:
    """将比较结果转为可 JSON 序列化结构。"""
    matches = []
    for match in comparison.get("matches", []):
        annotation = match.get("annotation")
        annotation_dict = {
            "start_frame": annotation.start_frame,
            "end_frame": annotation.end_frame,
            "label": annotation.label,
        }
        matches.append(
            {
                "annotation": annotation_dict,
                "pitch_value": match.get("pitch_value"),
                "yaw_value": match.get("yaw_value"),
                "expected_pitch": match.get("expected_pitch"),
                "expected_yaw": match.get("expected_yaw"),
                "pitch_axis_active": bool(match.get("pitch_axis_active", False)),
                "yaw_axis_active": bool(match.get("yaw_axis_active", False)),
                "pitch_match": bool(match.get("pitch_match", False)),
                "yaw_match": bool(match.get("yaw_match", False)),
                "is_match": bool(match.get("is_match", False)),
                "predicted_label": match.get("predicted_label"),
                "label_match": bool(match.get("label_match", False)),
            }
        )

    return {
        "frame_idx": comparison.get("frame_idx"),
        "video_id": comparison.get("video_id"),
        "angles": comparison.get("angles", {}),
        "adjusted_angles": comparison.get("adjusted_angles", {}),
        "matches": matches,
    }


def _calculate_axis_comparison_stats(comparisons: Dict[int, Dict]) -> Dict[str, float]:
    """计算分轴比较统计与比例。"""
    total_annotations = sum(len(comp.get("matches", [])) for comp in comparisons.values())

    yaw_eval_count = sum(
        sum(1 for m in comp.get("matches", []) if m.get("yaw_axis_active", False))
        for comp in comparisons.values()
    )
    yaw_match_count = sum(
        sum(
            1
            for m in comp.get("matches", [])
            if m.get("yaw_axis_active", False) and m.get("yaw_match", False)
        )
        for comp in comparisons.values()
    )

    pitch_eval_count = sum(
        sum(1 for m in comp.get("matches", []) if m.get("pitch_axis_active", False))
        for comp in comparisons.values()
    )
    pitch_match_count = sum(
        sum(
            1
            for m in comp.get("matches", [])
            if m.get("pitch_axis_active", False) and m.get("pitch_match", False)
        )
        for comp in comparisons.values()
    )

    yaw_match_rate = (yaw_match_count / yaw_eval_count * 100.0) if yaw_eval_count > 0 else 0.0
    pitch_match_rate = (
        pitch_match_count / pitch_eval_count * 100.0 if pitch_eval_count > 0 else 0.0
    )

    yaw_axis_eval_ratio = (
        yaw_eval_count / total_annotations * 100.0 if total_annotations > 0 else 0.0
    )
    pitch_axis_eval_ratio = (
        pitch_eval_count / total_annotations * 100.0 if total_annotations > 0 else 0.0
    )
    yaw_axis_match_ratio_in_all_annotations = (
        yaw_match_count / total_annotations * 100.0 if total_annotations > 0 else 0.0
    )
    pitch_axis_match_ratio_in_all_annotations = (
        pitch_match_count / total_annotations * 100.0 if total_annotations > 0 else 0.0
    )

    return {
        "total_annotations": total_annotations,
        "yaw_axis_eval_count": yaw_eval_count,
        "yaw_axis_match_count": yaw_match_count,
        "yaw_axis_match_rate": yaw_match_rate,
        "pitch_axis_eval_count": pitch_eval_count,
        "pitch_axis_match_count": pitch_match_count,
        "pitch_axis_match_rate": pitch_match_rate,
        "yaw_axis_eval_ratio": yaw_axis_eval_ratio,
        "pitch_axis_eval_ratio": pitch_axis_eval_ratio,
        "yaw_axis_match_ratio_in_all_annotations": yaw_axis_match_ratio_in_all_annotations,
        "pitch_axis_match_ratio_in_all_annotations": pitch_axis_match_ratio_in_all_annotations,
    }


def _calculate_discrete_comparison_stats(comparisons: Dict[int, Dict]) -> Dict[str, object]:
    """计算离散标签比较统计。"""
    total_annotations = sum(len(comp.get("matches", [])) for comp in comparisons.values())
    discrete_match_count = sum(
        sum(1 for m in comp.get("matches", []) if m.get("label_match", False))
        for comp in comparisons.values()
    )
    discrete_match_rate = (
        discrete_match_count / total_annotations * 100.0 if total_annotations > 0 else 0.0
    )

    by_direction_discrete = defaultdict(lambda: {"total": 0, "matched": 0, "rate": 0.0})
    for comp in comparisons.values():
        for m in comp.get("matches", []):
            label = str(m.get("annotation", {}).get("label", "unknown")).lower()
            by_direction_discrete[label]["total"] += 1
            if m.get("label_match", False):
                by_direction_discrete[label]["matched"] += 1

    for direction, stats in by_direction_discrete.items():
        total = stats["total"]
        stats["rate"] = (stats["matched"] / total * 100.0) if total > 0 else 0.0

    return {
        "discrete_match_count": discrete_match_count,
        "discrete_match_rate": discrete_match_rate,
        "by_direction_discrete": dict(by_direction_discrete),
    }


def _calculate_strict_comparison_stats(comparisons: Dict[int, Dict]) -> Dict[str, object]:
    """计算严格匹配统计（两轴联合 is_match）。"""
    total_annotations = sum(len(comp.get("matches", [])) for comp in comparisons.values())
    total_matches = sum(
        sum(1 for m in comp.get("matches", []) if m.get("is_match", False))
        for comp in comparisons.values()
    )
    match_rate = (total_matches / total_annotations * 100.0) if total_annotations > 0 else 0.0

    by_direction = defaultdict(lambda: {"total": 0, "matched": 0, "rate": 0.0})
    for comp in comparisons.values():
        for m in comp.get("matches", []):
            label = str(m.get("annotation", {}).get("label", "unknown")).lower()
            by_direction[label]["total"] += 1
            if m.get("is_match", False):
                by_direction[label]["matched"] += 1

    for direction, stats in by_direction.items():
        total = stats["total"]
        stats["rate"] = (stats["matched"] / total * 100.0) if total > 0 else 0.0

    return {
        "total_matches": total_matches,
        "match_rate": match_rate,
        "by_direction": dict(by_direction),
    }


def run_single_view_comparison(
    person_id: Optional[str] = None,
    env_jp: Optional[str] = None,
    view: str = "all",
    annotation_mode: str = "majority",
    threshold: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    cfg: Optional[DictConfig] = None,
):
    """运行单视角 SAM3D 与人工标注比较。"""
    cfg = cfg or load_config()
    env_mapping = get_env_mapping(cfg)
    sam3d_views = get_sam3d_views(cfg)

    person_id = person_id or str(cfg.defaults_run.person_id)
    env_jp = env_jp or str(cfg.defaults_run.env_jp)
    threshold = float(threshold if threshold is not None else cfg.defaults_run.threshold)
    if start_frame is None:
        start_frame = cfg.defaults_run.start_frame
    if end_frame is None:
        end_frame = cfg.defaults_run.end_frame

    view = view.lower()
    if view == "all":
        selected_views = list(sam3d_views)
    else:
        if view not in sam3d_views:
            raise ValueError(f"Unsupported view: {view}")
        selected_views = [view]

    env_en = env_mapping.get(env_jp, env_jp)
    video_id = f"{person_id}_{env_en}"

    annotation_file = get_annotation_file(cfg)
    if not annotation_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    annotation_mode = annotation_mode.lower().strip()
    if annotation_mode not in {"majority", "by_annotator"}:
        raise ValueError(f"Unsupported annotation_mode: {annotation_mode}")

    annotation_jobs: List[Tuple[str, Dict]] = []
    if annotation_mode == "majority":
        majority_annotations = load_majority_voted_annotations(annotation_file)
        annotation_jobs.append(("majority", majority_annotations))
    else:
        by_annotator = load_annotations_by_annotator(annotation_file)
        max_annotators = max((len(items) for items in by_annotator.values()), default=0)

        for annotator_idx in range(max_annotators):
            annotator_annotations: Dict[str, List] = {}
            for current_video_id, labels_by_annotator in by_annotator.items():
                if annotator_idx >= len(labels_by_annotator):
                    continue
                if labels_by_annotator[annotator_idx]:
                    annotator_annotations[current_video_id] = labels_by_annotator[annotator_idx]

            if annotator_annotations:
                annotation_jobs.append((f"annotator_{annotator_idx + 1}", annotator_annotations))

    if not annotation_jobs:
        raise ValueError(f"No annotations loaded for mode: {annotation_mode}")

    print("=" * 60)
    print("单视角 SAM3D 与人工标注比较")
    print("=" * 60)
    print("\n配置:")
    print(f"  Person ID: {person_id}")
    print(f"  Environment: {env_jp}")
    print(f"  视角: {', '.join(selected_views)}")
    print(f"  标注模式: {annotation_mode}")
    print(f"  匹配阈值: {threshold}°")
    print(f"  Frame范围: {start_frame} ~ {end_frame}")

    print("\n加载标注...")
    print(f"✓ 已加载 {len(annotation_jobs)} 组标注配置")

    for annotation_name, annotations in annotation_jobs:
        if video_id not in annotations:
            print(f"\n✗ 跳过 {annotation_name}: 未找到视频 {video_id} 的标注")
            continue

        analyzer = HeadPoseAnalyzer(annotation_dict=annotations)
        summary = []

        print(f"\n---- 标注来源: {annotation_name} ----")

        sam3d_output_root = get_single_view_output_root(cfg, threshold)
        if annotation_mode == "majority":
            output_root = sam3d_output_root / "majority"
        else:
            output_root = sam3d_output_root / "by_annotator" / annotation_name

        for current_view in selected_views:
            view_dir = get_sam3d_view_dir(cfg, person_id, env_jp, current_view)
            if not view_dir.exists():
                print(f"\n✗ 跳过 {current_view}: 目录不存在 {view_dir}")
                continue

            print(f"\n[{current_view}] 读取数据并计算角度...")
            angles_by_frame: Dict[int, Dict[str, float]] = {}

            for npz_path in sorted(view_dir.glob("*_sam3d_body.npz")):
                frame_idx = _parse_sam3d_frame_idx(npz_path.name)
                if frame_idx is None:
                    continue
                if start_frame is not None and frame_idx < start_frame:
                    continue
                if end_frame is not None and frame_idx > end_frame:
                    continue

                keypoints_3d = load_sam3d_keypoints(npz_path)
                if keypoints_3d is None:
                    continue

                head_kpts = extract_head_keypoints(keypoints_3d, analyzer.keypoint_indices)
                if head_kpts is None:
                    continue

                pitch, yaw = calculate_head_angles(head_kpts)
                angles_by_frame[frame_idx] = {
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                }

            front_baseline = _estimate_front_baseline_from_angles(
                analyzer,
                video_id,
                angles_by_frame,
            )
            baseline_angles = front_baseline["mean_angles"] if front_baseline else None

            comparisons: Dict[int, Dict] = {}
            for frame_idx, angles in sorted(angles_by_frame.items()):
                comparison = analyzer.compare_with_annotations(
                    video_id=video_id,
                    frame_idx=frame_idx,
                    angles=angles,
                    threshold_deg=threshold,
                    baseline_angles=baseline_angles,
                )
                if comparison:
                    comparisons[frame_idx] = _serialize_comparison(comparison)

            axis_stats = _calculate_axis_comparison_stats(comparisons)
            strict_stats = _calculate_strict_comparison_stats(comparisons)
            discrete_stats = _calculate_discrete_comparison_stats(comparisons)
            total_annotations = axis_stats["total_annotations"]
            yaw_eval_count = axis_stats["yaw_axis_eval_count"]
            yaw_match_count = axis_stats["yaw_axis_match_count"]
            yaw_match_rate = axis_stats["yaw_axis_match_rate"]
            pitch_eval_count = axis_stats["pitch_axis_eval_count"]
            pitch_match_count = axis_stats["pitch_axis_match_count"]
            pitch_match_rate = axis_stats["pitch_axis_match_rate"]
            annotated_frame_ratio = (
                len(comparisons) / len(angles_by_frame) * 100.0
                if len(angles_by_frame) > 0
                else 0.0
            )

            output_dir = get_sam3d_output_dir(
                cfg,
                person_id,
                env_en,
                current_view,
                output_root=output_root,
            )
            output_file = output_dir / "result.json"

            result = {
                "source": "sam3d_view",
                "annotation_mode": annotation_mode,
                "annotation_source": annotation_name,
                "person_id": person_id,
                "env_jp": env_jp,
                "env_en": env_en,
                "view": current_view,
                "video_id": video_id,
                "threshold_deg": threshold,
                "total_frames": len(angles_by_frame),
                "annotated_frames": len(comparisons),
                "annotated_frame_ratio": annotated_frame_ratio,
                "total_annotations": total_annotations,
                "total_matches": strict_stats["total_matches"],
                "match_rate": strict_stats["match_rate"],
                "yaw_axis_eval_count": yaw_eval_count,
                "yaw_axis_match_count": yaw_match_count,
                "yaw_axis_match_rate": yaw_match_rate,
                "yaw_axis_eval_ratio": axis_stats["yaw_axis_eval_ratio"],
                "yaw_axis_match_ratio_in_all_annotations": axis_stats[
                    "yaw_axis_match_ratio_in_all_annotations"
                ],
                "pitch_axis_eval_count": pitch_eval_count,
                "pitch_axis_match_count": pitch_match_count,
                "pitch_axis_match_rate": pitch_match_rate,
                "pitch_axis_eval_ratio": axis_stats["pitch_axis_eval_ratio"],
                "pitch_axis_match_ratio_in_all_annotations": axis_stats[
                    "pitch_axis_match_ratio_in_all_annotations"
                ],
                "discrete_match_count": discrete_stats["discrete_match_count"],
                "discrete_match_rate": discrete_stats["discrete_match_rate"],
                "by_direction": strict_stats["by_direction"],
                "by_direction_discrete": discrete_stats["by_direction_discrete"],
                "angles": {str(k): v for k, v in angles_by_frame.items()},
                "comparisons": {str(k): v for k, v in comparisons.items()},
            }

            with output_file.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)

            print(
                f"✓ [{current_view}] 帧数={len(angles_by_frame)}, 标注数={total_annotations}, "
                f"有标注帧占比={annotated_frame_ratio:.2f}%, "
                f"Yaw分轴={yaw_match_rate:.2f}%, Pitch分轴={pitch_match_rate:.2f}%, "
                f"离散匹配={discrete_stats['discrete_match_rate']:.2f}%"
            )

            summary.append(
                {
                    "annotation_mode": annotation_mode,
                    "annotation_source": annotation_name,
                    "view": current_view,
                    "total_frames": len(angles_by_frame),
                    "annotated_frames": len(comparisons),
                    "annotated_frame_ratio": annotated_frame_ratio,
                    "total_annotations": total_annotations,
                    "total_matches": strict_stats["total_matches"],
                    "match_rate": strict_stats["match_rate"],
                    "yaw_axis_eval_count": yaw_eval_count,
                    "yaw_axis_match_count": yaw_match_count,
                    "yaw_axis_match_rate": yaw_match_rate,
                    "yaw_axis_eval_ratio": axis_stats["yaw_axis_eval_ratio"],
                    "yaw_axis_match_ratio_in_all_annotations": axis_stats[
                        "yaw_axis_match_ratio_in_all_annotations"
                    ],
                    "pitch_axis_eval_count": pitch_eval_count,
                    "pitch_axis_match_count": pitch_match_count,
                    "pitch_axis_match_rate": pitch_match_rate,
                    "pitch_axis_eval_ratio": axis_stats["pitch_axis_eval_ratio"],
                    "pitch_axis_match_ratio_in_all_annotations": axis_stats[
                        "pitch_axis_match_ratio_in_all_annotations"
                    ],
                    "discrete_match_count": discrete_stats["discrete_match_count"],
                    "discrete_match_rate": discrete_stats["discrete_match_rate"],
                    "by_direction": strict_stats["by_direction"],
                    "by_direction_discrete": discrete_stats["by_direction_discrete"],
                    "output_file": str(output_file),
                }
            )

        if summary:
            summary_root = output_root / person_id / env_en
            summary_root.mkdir(parents=True, exist_ok=True)
            summary_file = summary_root / "summary.json"
            with summary_file.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            print(f"\n✓ [{annotation_name}] 汇总已保存: {summary_file}")
        else:
            print(f"\n✗ [{annotation_name}] 没有生成任何结果")

    print("\n" + "=" * 60)


def run_single_view_comparison_all(
    view: str = "all",
    annotation_mode: str = "majority",
    threshold: Optional[float] = None,
    cfg: Optional[DictConfig] = None,
):
    """批量运行 single_view：遍历所有可用的人物与环境。"""
    cfg = cfg or load_config()
    sam3d_views = get_sam3d_views(cfg)
    threshold = float(threshold if threshold is not None else cfg.defaults_run.threshold)

    view = view.lower()
    if view == "all":
        selected_views = list(sam3d_views)
    else:
        if view not in sam3d_views:
            raise ValueError(f"Unsupported view: {view}")
        selected_views = [view]

    pairs = _get_all_sam3d_person_env_pairs(selected_views, cfg)
    print("=" * 80)
    print("single_view 全量模式：遍历所有 person/env")
    print("=" * 80)
    print(f"视角: {', '.join(selected_views)}")
    print(f"标注模式: {annotation_mode}")
    print(f"匹配阈值: {threshold}°")
    print(f"找到 {len(pairs)} 个人物-环境组合")

    if not pairs:
        print("✗ 未找到可用的单视角 SAM3D 数据")
        return

    for idx, (person_id, env_jp) in enumerate(pairs, 1):
        print(f"\n[{idx}/{len(pairs)}] {person_id} - {env_jp}")
        run_single_view_comparison(
            person_id=person_id,
            env_jp=env_jp,
            view=view,
            annotation_mode=annotation_mode,
            threshold=threshold,
            cfg=cfg,
        )


if __name__ == "__main__":
    run_comparison()

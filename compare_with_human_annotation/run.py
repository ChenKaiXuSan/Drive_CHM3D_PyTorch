#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
与人工标注比较的运行脚本
使用 config.py 中定义的默认路径
"""
import json
from typing import Dict, List, Optional, Tuple

from pathlib import Path

import numpy as np

from .main import HeadPoseAnalyzer
from .load import (
    get_unlabeled_frame_indices,
    load_annotations_by_annotator,
    load_head_movement_annotations,
    load_majority_voted_annotations,
    load_sam3d_keypoints,
)
from .angle_calculator import calculate_head_angles, extract_head_keypoints
from .config import (
    OUTPUT_ROOT_SAM3D_VIEWS,
    SAM3D_BODY_RESULTS_ROOT,
    SAM3D_VIEWS,
    get_annotation_file,
    get_fused_dir,
    get_output_dir,
    get_sam3d_output_dir,
    get_sam3d_view_dir,
    DEFAULT_PERSON_ID,
    DEFAULT_ENV_JP,
    DEFAULT_START_FRAME,
    DEFAULT_END_FRAME,
    DEFAULT_THRESHOLD,
    ENVIRONMENTS,
)


def _get_all_sam3d_person_env_pairs(selected_views: List[str]) -> List[Tuple[str, str]]:
    """获取包含单视角 SAM3D 结果的全部人物-环境组合。"""
    pairs: List[Tuple[str, str]] = []

    if not SAM3D_BODY_RESULTS_ROOT.exists():
        return pairs

    for person_dir in sorted(SAM3D_BODY_RESULTS_ROOT.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name

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
                pairs.append((person_id, env_dir.name))

    return pairs


def run_comparison(
    person_id: Optional[str] = None,
    env_jp: Optional[str] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    threshold: Optional[float] = None,
):
    """
    运行头部姿态与标注比较
    
    Args:
        person_id: 人物ID (默认: config.DEFAULT_PERSON_ID)
        env_jp: 环境日文名称 (默认: config.DEFAULT_ENV_JP)
        start_frame: 起始帧 (默认: config.DEFAULT_START_FRAME)
        end_frame: 结束帧 (默认: config.DEFAULT_END_FRAME)
        threshold: 匹配阈值度数 (默认: config.DEFAULT_THRESHOLD)
    """
    # 使用默认值
    person_id = person_id or DEFAULT_PERSON_ID
    env_jp = env_jp or DEFAULT_ENV_JP
    start_frame = start_frame or DEFAULT_START_FRAME
    end_frame = end_frame or DEFAULT_END_FRAME
    threshold = threshold or DEFAULT_THRESHOLD
    
    print("=" * 60)
    print("头部姿态与人工标注比较")
    print("=" * 60)
    print("\n配置:")
    print(f"  Person ID: {person_id}")
    print(f"  Environment: {env_jp}")
    print(f"  Frame范围: {start_frame} ~ {end_frame}")
    print(f"  匹配阈值: {threshold}°")
    
    # 获取路径
    annotation_file = get_annotation_file()
    fused_dir = get_fused_dir(person_id, env_jp)
    env_en = ENVIRONMENTS.get(env_jp, env_jp)
    output_dir = get_output_dir(person_id, env_en)
    
    print("\n路径:")
    print(f"  标注文件: {annotation_file}")
    print(f"  融合数据: {fused_dir}")
    print(f"  输出目录: {output_dir}")
    
    # 检查文件存在性
    if not annotation_file.exists():
        print(f"\n✗ 错误: 标注文件不存在 {annotation_file}")
        return
    
    if not fused_dir.exists():
        print(f"\n✗ 错误: 融合数据目录不存在 {fused_dir}")
        return
    
    # 加载标注
    print("\n加载标注...")
    annotations = load_head_movement_annotations(annotation_file)
    print(f"✓ 已加载 {len(annotations)} 个视频的标注")
    
    # 创建分析器
    print("\n创建分析器...")
    analyzer = HeadPoseAnalyzer(annotation_dict=annotations)
    print("✓ 分析器已创建")

    video_id = f"{person_id}_{env_en}"

    # 估计 front baseline（使用未被任何标注覆盖的帧）
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
            f"Yaw={mean_angles['yaw']:.2f}°, Roll={mean_angles['roll']:.2f}°"
        )
    
    # 分析并比较
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
    
    # 统计匹配率
    if comparisons:
        total_matches = sum(len(comp["matches"]) for comp in comparisons.values())
        successful_matches = sum(sum(1 for m in comp["matches"] if m["is_match"]) 
                               for comp in comparisons.values())
        match_rate = (successful_matches / total_matches * 100) if total_matches > 0 else 0
        
        print("\n匹配统计:")
        print(f"  总标注数: {total_matches}")
        print(f"  匹配数: {successful_matches}")
        print(f"  匹配率: {match_rate:.1f}%")
        
        # 显示前5帧的详细结果
        print("\n前5帧的详细结果:")
        for frame_idx, comparison in sorted(list(comparisons.items())[:5]):
            print(f"\n  Frame {frame_idx}:")
            for match in comparison["matches"]:
                status = "✓" if match["is_match"] else "✗"
                print(f"    {status} {match['annotation'].label}: "
                      f"Pitch={match['pitch_value']:6.1f}°, "
                      f"Yaw={match['yaw_value']:6.1f}°")
    else:
        print("\n✗ 没有找到有标注的帧")
    
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

    candidates: List[Tuple[float, int, Dict[str, float]]] = []
    for frame_idx in unlabeled_frames:
        angles = angles_by_frame.get(frame_idx)
        if angles is None:
            continue
        score = analyzer._front_score(
            (angles["pitch"], angles["yaw"], angles["roll"])
        )
        candidates.append((score, frame_idx, angles))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    selected_count = max(
        min_selected_frames,
        int(len(candidates) * selection_ratio),
    )
    selected_count = min(selected_count, max_selected_frames, len(candidates))
    selected = candidates[:selected_count]

    mean_pitch = float(np.mean([item[2]["pitch"] for item in selected]))
    mean_yaw = float(np.mean([item[2]["yaw"] for item in selected]))
    mean_roll = float(np.mean([item[2]["roll"] for item in selected]))

    return {
        "video_id": video_id,
        "frame_count": selected_count,
        "candidate_frame_count": len(candidates),
        "selection_ratio": selection_ratio,
        "frame_indices": [item[1] for item in selected],
        "mean_angles": {
            "pitch": mean_pitch,
            "yaw": mean_yaw,
            "roll": mean_roll,
        },
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
                "is_match": bool(match.get("is_match", False)),
            }
        )

    return {
        "frame_idx": comparison.get("frame_idx"),
        "video_id": comparison.get("video_id"),
        "angles": comparison.get("angles", {}),
        "adjusted_angles": comparison.get("adjusted_angles", {}),
        "matches": matches,
    }


def run_single_view_comparison(
    person_id: Optional[str] = None,
    env_jp: Optional[str] = None,
    view: str = "all",
    annotation_mode: str = "majority",
    threshold: Optional[float] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
):
    """
    运行单视角 SAM3D 与人工标注比较。

    Args:
        person_id: 人物 ID（默认配置值）
        env_jp: 环境（日文，默认配置值）
        view: 视角（front/left/right/all）
        annotation_mode: 标注模式（majority/by_annotator）
        threshold: 匹配阈值（度）
        start_frame: 起始帧（可选）
        end_frame: 结束帧（可选）
    """
    person_id = person_id or DEFAULT_PERSON_ID
    env_jp = env_jp or DEFAULT_ENV_JP
    threshold = threshold or DEFAULT_THRESHOLD

    view = view.lower()
    if view == "all":
        selected_views = list(SAM3D_VIEWS)
    else:
        if view not in SAM3D_VIEWS:
            raise ValueError(f"Unsupported view: {view}")
        selected_views = [view]

    env_en = ENVIRONMENTS.get(env_jp, env_jp)
    video_id = f"{person_id}_{env_en}"

    annotation_file = get_annotation_file()
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
        if by_annotator:
            max_annotators = max(len(items) for items in by_annotator.values())
        else:
            max_annotators = 0

        for annotator_idx in range(max_annotators):
            annotator_annotations: Dict[str, List] = {}
            for current_video_id, labels_by_annotator in by_annotator.items():
                if annotator_idx >= len(labels_by_annotator):
                    continue
                if labels_by_annotator[annotator_idx]:
                    annotator_annotations[current_video_id] = labels_by_annotator[annotator_idx]

            if annotator_annotations:
                annotation_jobs.append(
                    (f"annotator_{annotator_idx + 1}", annotator_annotations)
                )

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

        if annotation_mode == "majority":
            output_root = OUTPUT_ROOT_SAM3D_VIEWS / "majority"
        else:
            output_root = OUTPUT_ROOT_SAM3D_VIEWS / "by_annotator" / annotation_name

        for current_view in selected_views:
            view_dir = get_sam3d_view_dir(person_id, env_jp, current_view)
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

                pitch, yaw, roll = calculate_head_angles(head_kpts)
                angles_by_frame[frame_idx] = {
                    "pitch": float(pitch),
                    "yaw": float(yaw),
                    "roll": float(roll),
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

            total_annotations = sum(len(comp["matches"]) for comp in comparisons.values())
            total_matches = sum(
                sum(1 for match in comp["matches"] if match["is_match"])
                for comp in comparisons.values()
            )
            match_rate = (
                total_matches / total_annotations * 100
                if total_annotations > 0
                else 0.0
            )

            output_dir = get_sam3d_output_dir(
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
                "front_baseline": front_baseline,
                "total_frames": len(angles_by_frame),
                "annotated_frames": len(comparisons),
                "total_annotations": total_annotations,
                "total_matches": total_matches,
                "match_rate": match_rate,
                "angles": {str(k): v for k, v in angles_by_frame.items()},
                "comparisons": {str(k): v for k, v in comparisons.items()},
            }

            with output_file.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)

            print(
                f"✓ [{current_view}] 帧数={len(angles_by_frame)}, 标注数={total_annotations}, "
                f"匹配率={match_rate:.2f}%"
            )

            summary.append(
                {
                    "annotation_mode": annotation_mode,
                    "annotation_source": annotation_name,
                    "view": current_view,
                    "total_frames": len(angles_by_frame),
                    "annotated_frames": len(comparisons),
                    "total_annotations": total_annotations,
                    "total_matches": total_matches,
                    "match_rate": match_rate,
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


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    person_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERSON_ID
    env_jp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ENV_JP
    
    run_comparison(person_id=person_id, env_jp=env_jp)


def run_single_view_comparison_all(
    view: str = "all",
    annotation_mode: str = "majority",
    threshold: Optional[float] = None,
):
    """批量运行 single_view：遍历所有可用的人物与环境。"""
    threshold = threshold or DEFAULT_THRESHOLD

    view = view.lower()
    if view == "all":
        selected_views = list(SAM3D_VIEWS)
    else:
        if view not in SAM3D_VIEWS:
            raise ValueError(f"Unsupported view: {view}")
        selected_views = [view]

    pairs = _get_all_sam3d_person_env_pairs(selected_views)
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
        )

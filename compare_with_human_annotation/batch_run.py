#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
批量处理所有人物和环境，生成论文级别的汇总报告
"""

from pathlib import Path
import json
from collections import defaultdict
from typing import Optional

from omegaconf import DictConfig

from .head_pose_analyzer import HeadPoseAnalyzer
from .load import (
    load_head_movement_annotations,
    load_majority_voted_annotations,
    load_annotations_by_annotator,
)
from .hydra_utils import (
    get_annotation_file,
    get_env_mapping,
    get_fused_dir,
    get_fused_root,
    get_output_dir,
    get_output_source_root,
    load_config,
)


def _calculate_axis_comparison_stats(comparisons) -> dict:
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
        "yaw_axis_eval_ratio": yaw_axis_eval_ratio,
        "yaw_axis_match_ratio_in_all_annotations": yaw_axis_match_ratio_in_all_annotations,
        "pitch_axis_eval_count": pitch_eval_count,
        "pitch_axis_match_count": pitch_match_count,
        "pitch_axis_match_rate": pitch_match_rate,
        "pitch_axis_eval_ratio": pitch_axis_eval_ratio,
        "pitch_axis_match_ratio_in_all_annotations": pitch_axis_match_ratio_in_all_annotations,
    }


def get_all_person_env_pairs(smoothed: bool = False, cfg: Optional[DictConfig] = None):
    """获取所有可用的 person 和 environment 组合。"""
    cfg = cfg or load_config()
    fused_root = get_fused_root(cfg)
    fused_subdir = str(cfg.fused.subdir_smoothed if smoothed else cfg.fused.subdir_raw)
    pairs = []

    for person_dir in sorted(fused_root.iterdir()):
        if not person_dir.is_dir():
            continue
        person_id = person_dir.name

        for env_dir in sorted(person_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            env_jp = env_dir.name

            fused_dir = env_dir / fused_subdir
            if fused_dir.exists() and any(fused_dir.glob("frame_*_fused.npy")):
                pairs.append((person_id, env_jp))

    return pairs


def run_single_comparison(
    analyzer,
    person_id,
    env_jp,
    threshold_deg: Optional[float] = None,
    smoothed: bool = False,
    cfg: Optional[DictConfig] = None,
):
    """
    运行单个 person-env 的比较

    Returns:
        {
            'person_id': str,
            'env_jp': str,
            'env_en': str,
            'total_frames': int,
            'annotated_frames': int,
            'total_annotations': int,
            'total_matches': int,
            'match_rate': float,
            'by_direction': dict
        }
    """
    cfg = cfg or load_config()
    env_mapping = get_env_mapping(cfg)
    threshold_deg = float(
        threshold_deg if threshold_deg is not None else cfg.defaults_run.threshold
    )

    env_en = env_mapping.get(env_jp)
    if env_en is None:
        env_en = env_jp
    env_en = str(env_en)
    fused_dir = get_fused_dir(cfg, person_id, env_jp, smoothed=smoothed)
    video_id = f"{person_id}_{env_en}"

    front_baseline = analyzer.estimate_front_baseline(
        video_id=video_id,
        fused_dir=fused_dir,
    )
    baseline_angles = front_baseline["mean_angles"] if front_baseline else None

    # 分析
    results = analyzer.analyze_sequence_with_annotations(
        video_id=video_id,
        fused_dir=fused_dir,
        threshold_deg=threshold_deg,
        baseline_angles=baseline_angles,
    )

    angles = results["angles"]
    comparisons = results["comparisons"]

    # 统计
    total_frames = len(angles)
    annotated_frames = len(comparisons)
    annotated_frame_ratio = (
        (annotated_frames / total_frames * 100.0) if total_frames > 0 else 0.0
    )

    # 按方向统计
    by_direction = defaultdict(lambda: {"total": 0, "matched": 0, "rate": 0.0})
    total_annotations = 0
    total_matches = 0

    for comparison in comparisons.values():
        for match in comparison["matches"]:
            label = match["annotation"].label.lower()
            if label not in by_direction:
                by_direction[label] = {"total": 0, "matched": 0, "rate": 0.0}

            by_direction[label]["total"] += 1
            total_annotations += 1

            if match["is_match"]:
                by_direction[label]["matched"] += 1
                total_matches += 1

    # 计算匹配率
    match_rate = (
        (total_matches / total_annotations * 100) if total_annotations > 0 else 0
    )

    axis_stats = _calculate_axis_comparison_stats(comparisons)

    # 计算每个方向的匹配率
    for direction in by_direction:
        total = by_direction[direction]["total"]
        matched = by_direction[direction]["matched"]
        by_direction[direction]["rate"] = (matched / total * 100) if total > 0 else 0

    return {
        "person_id": person_id,
        "env_jp": env_jp,
        "env_en": env_en,
        "front_baseline": front_baseline,
        "baseline": baseline_angles,
        "total_frames": total_frames,
        "annotated_frames": annotated_frames,
        "annotated_frame_ratio": annotated_frame_ratio,
        "total_annotations": total_annotations,
        "total_matches": total_matches,
        "match_rate": match_rate,
        "yaw_axis_eval_count": axis_stats["yaw_axis_eval_count"],
        "yaw_axis_match_count": axis_stats["yaw_axis_match_count"],
        "yaw_axis_match_rate": axis_stats["yaw_axis_match_rate"],
        "yaw_axis_eval_ratio": axis_stats["yaw_axis_eval_ratio"],
        "yaw_axis_match_ratio_in_all_annotations": axis_stats[
            "yaw_axis_match_ratio_in_all_annotations"
        ],
        "pitch_axis_eval_count": axis_stats["pitch_axis_eval_count"],
        "pitch_axis_match_count": axis_stats["pitch_axis_match_count"],
        "pitch_axis_match_rate": axis_stats["pitch_axis_match_rate"],
        "pitch_axis_eval_ratio": axis_stats["pitch_axis_eval_ratio"],
        "pitch_axis_match_ratio_in_all_annotations": axis_stats[
            "pitch_axis_match_ratio_in_all_annotations"
        ],
        "by_direction": dict(by_direction),
    }


def generate_paper_report(
    all_results,
    output_file,
    threshold_deg=5.0,
    report_title="头部姿态估计与人工标注比较 - 论文报告",
):
    """
    生成论文级别的汇总报告

    Args:
        all_results: 所有比较结果的列表
        output_file: 输出文件路径
        threshold_deg: 判断是否匹配的阈值（度）
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        # 标题
        f.write("=" * 80 + "\n")
        f.write(f"{report_title}\n")
        f.write("=" * 80 + "\n\n")

        # 摘要统计
        f.write("## 1. 总体统计\n\n")

        total_frames = sum(r["total_frames"] for r in all_results)
        total_annotated = sum(r["annotated_frames"] for r in all_results)
        total_annotations = sum(r["total_annotations"] for r in all_results)
        total_matches = sum(r["total_matches"] for r in all_results)
        overall_rate = (
            (total_matches / total_annotations * 100) if total_annotations > 0 else 0
        )
        overall_annotated_frame_ratio = (
            (total_annotated / total_frames * 100) if total_frames > 0 else 0
        )

        total_yaw_eval = sum(r.get("yaw_axis_eval_count", 0) for r in all_results)
        total_yaw_match = sum(r.get("yaw_axis_match_count", 0) for r in all_results)
        total_pitch_eval = sum(r.get("pitch_axis_eval_count", 0) for r in all_results)
        total_pitch_match = sum(r.get("pitch_axis_match_count", 0) for r in all_results)

        yaw_axis_match_rate = (
            (total_yaw_match / total_yaw_eval * 100) if total_yaw_eval > 0 else 0
        )
        pitch_axis_match_rate = (
            (total_pitch_match / total_pitch_eval * 100) if total_pitch_eval > 0 else 0
        )

        f.write(f"数据集规模：{len(all_results)} 个人-环境组合\n")
        f.write(f"总帧数：{total_frames:,}\n")
        f.write(
            f"有标注的帧：{total_annotated:,} ({100 * total_annotated / total_frames:.1f}%)\n"
        )
        f.write(f"有标注帧占比：{overall_annotated_frame_ratio:.1f}%\n")
        f.write(f"总标注数：{total_annotations:,}\n")
        f.write(f"匹配数：{total_matches:,}\n")
        f.write(f"**总体匹配率：{overall_rate:.1f}%**\n\n")
        f.write(
            f"Yaw 分轴匹配率：{yaw_axis_match_rate:.1f}% ({total_yaw_match}/{total_yaw_eval})\n"
        )
        f.write(
            f"Pitch 分轴匹配率：{pitch_axis_match_rate:.1f}% ({total_pitch_match}/{total_pitch_eval})\n\n"
        )

        # 按环境分类统计
        f.write("## 2. 按环境分类统计\n\n")

        env_stats = defaultdict(
            lambda: {
                "count": 0,
                "total_annotations": 0,
                "total_matches": 0,
            }
        )

        for result in all_results:
            env_en = result["env_en"]
            env_stats[env_en]["count"] += 1
            env_stats[env_en]["total_annotations"] += result["total_annotations"]
            env_stats[env_en]["total_matches"] += result["total_matches"]

        f.write("| 环境 | 人数 | 标注数 | 匹配数 | 匹配率 |\n")
        f.write("|------|------|--------|--------|--------|\n")

        for env_en in sorted(env_stats.keys()):
            stat = env_stats[env_en]
            rate = (
                (stat["total_matches"] / stat["total_annotations"] * 100)
                if stat["total_annotations"] > 0
                else 0
            )
            f.write(
                f"| {env_en:12s} | {stat['count']:4d} | {stat['total_annotations']:6d} | {stat['total_matches']:6d} | {rate:6.1f}% |\n"
            )

        f.write("\n")

        # baseline统计
        f.write("## 3. Front Baseline 统计\n\n")

        baseline_rows = [
            r
            for r in all_results
            if r.get("front_baseline") and r["front_baseline"].get("mean_angles")
        ]
        if baseline_rows:
            baseline_pitch = [
                r["front_baseline"]["mean_angles"]["pitch"] for r in baseline_rows
            ]
            baseline_yaw = [
                r["front_baseline"]["mean_angles"]["yaw"] for r in baseline_rows
            ]
            selected_counts = [
                r["front_baseline"]["frame_count"] for r in baseline_rows
            ]

            f.write(f"可用 baseline 数量：{len(baseline_rows)} / {len(all_results)}\n")
            f.write(
                f"平均 baseline angles：Pitch={sum(baseline_pitch) / len(baseline_pitch):.2f}°, "
                f"Yaw={sum(baseline_yaw) / len(baseline_yaw):.2f}°\n"
            )
            f.write(
                f"baseline 选帧数范围：{min(selected_counts)} ~ {max(selected_counts)}\n"
            )
        else:
            f.write("没有可用的 baseline 统计。\n")

        f.write("\n")

        # 按方向分类统计
        f.write("## 4. 按头部转动方向分类统计\n\n")

        direction_stats = defaultdict(
            lambda: {
                "total": 0,
                "matched": 0,
            }
        )

        for result in all_results:
            for direction, stat in result["by_direction"].items():
                direction_stats[direction]["total"] += stat["total"]
                direction_stats[direction]["matched"] += stat["matched"]

        f.write("| 方向 | 标注数 | 匹配数 | 匹配率 |\n")
        f.write("|------|--------|--------|--------|\n")

        for direction in sorted(direction_stats.keys()):
            stat = direction_stats[direction]
            rate = (stat["matched"] / stat["total"] * 100) if stat["total"] > 0 else 0
            f.write(
                f"| {direction:8s} | {stat['total']:6d} | {stat['matched']:6d} | {rate:6.1f}% |\n"
            )

        f.write("\n")

        # 详细结果表
        f.write("## 5. 详细结果表（按人物和环境）\n\n")

        f.write("| 人物 ID | 环境 | 总帧数 | 有标注 | 标注数 | 匹配数 | 匹配率 |\n")
        f.write("|---------|------|--------|--------|--------|--------|--------|\n")

        for result in sorted(all_results, key=lambda x: (x["person_id"], x["env_en"])):
            f.write(
                f"| {result['person_id']:6s} | {result['env_en']:12s} | "
                f"{result['total_frames']:6d} | {result['annotated_frames']:6d} | "
                f"{result['total_annotations']:6d} | {result['total_matches']:6d} | "
                f"{result['match_rate']:6.1f}% |\n"
            )

        f.write("\n")

        # 讨论
        f.write("## 6. 讨论\n\n")
        f.write(
            f"本研究对 {len(all_results)} 个人-环境组合进行了头部姿态估计与人工标注的比较。\n"
        )
        f.write(
            f"在阈值 τ={threshold_deg:g}° 的条件下，总体匹配率为 {overall_rate:.1f}%。\n\n"
        )

        f.write("### 按环境分析\n\n")
        for env_en in sorted(env_stats.keys()):
            stat = env_stats[env_en]
            rate = (
                (stat["total_matches"] / stat["total_annotations"] * 100)
                if stat["total_annotations"] > 0
                else 0
            )
            f.write(
                f"- **{env_en}**：{rate:.1f}% (基于 {stat['total_annotations']} 个标注)\n"
            )

        f.write("\n### 按方向分析\n\n")
        for direction in sorted(direction_stats.keys()):
            stat = direction_stats[direction]
            rate = (stat["matched"] / stat["total"] * 100) if stat["total"] > 0 else 0
            f.write(f"- **{direction}**：{rate:.1f}% (基于 {stat['total']} 个标注)\n")

        f.write("\n")
        f.write("=" * 80 + "\n")


def _run_batch_comparison(
    cfg: DictConfig,
    annotations,
    output_root,
    threshold_deg: Optional[float] = None,
    report_title="头部姿态估计与人工标注比较 - 论文报告",
    smoothed: bool = False,
):
    """执行批量比较并写入指定输出目录。"""
    threshold_deg = float(
        threshold_deg if threshold_deg is not None else cfg.defaults_run.threshold
    )
    env_mapping = get_env_mapping(cfg)

    pairs = get_all_person_env_pairs(smoothed=smoothed, cfg=cfg)
    print(f"\n找到 {len(pairs)} 个人-环境组合")

    analyzer = HeadPoseAnalyzer(annotation_dict=annotations)

    print("\n开始批量比较...\n")
    all_results = []

    for idx, (person_id, env_jp) in enumerate(pairs, 1):
        env_en = env_mapping.get(env_jp)
        if env_en is None:
            env_en = env_jp
        env_en = str(env_en)
        print(
            f"[{idx}/{len(pairs)}] {person_id} - {env_jp} ({env_en})...",
            end=" ",
            flush=True,
        )

        try:
            result = run_single_comparison(
                analyzer,
                person_id,
                env_jp,
                threshold_deg=threshold_deg,
                smoothed=smoothed,
                cfg=cfg,
            )
            all_results.append(result)

            output_dir = get_output_dir(person_id, env_en, output_root=output_root)
            result_file = output_dir / "result.json"
            with result_file.open("w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"✓ {result['match_rate']:.1f}%")
        except Exception as e:
            print(f"✗ 错误: {e}")

    print("\n" + "=" * 80)

    print("\n生成论文级汇总报告...")
    report_file = output_root / "paper_report.txt"
    generate_paper_report(
        all_results,
        report_file,
        threshold_deg=threshold_deg,
        report_title=report_title,
    )
    print(f"✓ 报告已保存: {report_file}")

    summary_file = output_root / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ JSON 汇总已保存: {summary_file}")

    print("\n" + "=" * 80)
    print("批量比较完成！")
    print("=" * 80)


def run_batch_comparison(
    threshold_deg: Optional[float] = None,
    cfg: Optional[DictConfig] = None,
):
    """
    批量比较所有人物和环境，生成汇总报告

    Args:
        threshold_deg: 判断是否匹配的阈值（度）
    """
    cfg = cfg or load_config()
    threshold_deg = float(
        threshold_deg if threshold_deg is not None else cfg.defaults_run.threshold
    )

    print("=" * 80)
    print("批量比较所有人物和环境（raw + smoothed）")
    print("=" * 80)
    print(f"匹配阈值: {threshold_deg:g}°")

    print("\n加载标注...")
    annotation_file = get_annotation_file(cfg)
    annotations = load_head_movement_annotations(annotation_file)
    print(f"✓ 已加载 {len(annotations)} 个视频的标注")

    for smoothed in (False, True):
        source_name = "smoothed" if smoothed else "raw"
        source_label = "平滑" if smoothed else "未平滑"
        output_root = get_output_source_root(cfg, smoothed=smoothed)

        print(f"\n{'=' * 80}")
        print(f"开始比较来源: {source_label} ({source_name})")
        print(f"输出目录: {output_root}")
        print(f"{'=' * 80}")

        _run_batch_comparison(
            cfg=cfg,
            annotations=annotations,
            output_root=output_root,
            threshold_deg=threshold_deg,
            report_title=f"头部姿态估计与人工标注比较 - 论文报告（{source_label}）",
            smoothed=smoothed,
        )


def run_batch_comparison_majority_vote(
    threshold_deg: Optional[float] = None,
    cfg: Optional[DictConfig] = None,
):
    """
    先对 3 位标注者做多数投票，再与模型比较，结果输出到独立目录。
    """
    cfg = cfg or load_config()
    threshold_deg = float(
        threshold_deg if threshold_deg is not None else cfg.defaults_run.threshold
    )

    print("=" * 80)
    print("多数投票后批量比较所有人物和环境（raw + smoothed）")
    print("=" * 80)
    print(f"匹配阈值: {threshold_deg:g}°")

    print("\n加载标注并执行多数投票...")
    annotation_file = get_annotation_file(cfg)
    annotations = load_majority_voted_annotations(annotation_file)
    print(f"✓ 已生成 {len(annotations)} 个视频的多数投票标注")

    for smoothed in (False, True):
        source_label = "平滑" if smoothed else "未平滑"
        output_root = get_output_source_root(cfg, smoothed=smoothed) / "majority"

        print(f"\n{'=' * 80}")
        print(f"开始比较来源: {source_label}")
        print(f"输出目录: {output_root}")
        print(f"{'=' * 80}")

        _run_batch_comparison(
            cfg=cfg,
            annotations=annotations,
            output_root=output_root,
            threshold_deg=threshold_deg,
            report_title=f"头部姿态估计与人工标注比较 - 论文报告（3位标注者多数投票，{source_label}）",
            smoothed=smoothed,
        )


def run_batch_comparison_by_annotator(
    threshold_deg: Optional[float] = None,
    cfg: Optional[DictConfig] = None,
):
    """
    分别与3位标注者的标注进行比较，生成3份独立结果
    """
    cfg = cfg or load_config()
    threshold_deg = float(
        threshold_deg if threshold_deg is not None else cfg.defaults_run.threshold
    )

    print("=" * 80)
    print("分别与每位标注者比较（raw + smoothed）")
    print("=" * 80)
    print(f"匹配阈值: {threshold_deg:g}°")

    print("\n加载按标注者分组的标注...")
    annotation_file = get_annotation_file(cfg)
    annotations_by_annotator = load_annotations_by_annotator(annotation_file)
    print(f"✓ 已加载 {len(annotations_by_annotator)} 个视频的标注")

    first_video_annotations = (
        next(iter(annotations_by_annotator.values()))
        if annotations_by_annotator
        else []
    )
    num_annotators = len(first_video_annotations)
    print(f"✓ 检测到 {num_annotators} 位标注者")

    for smoothed in (False, True):
        source_label = "平滑" if smoothed else "未平滑"
        source_root = get_output_source_root(cfg, smoothed=smoothed) / "by_annotator"
        print(f"\n{'=' * 80}")
        print(f"开始比较来源: {source_label}")
        print(f"输出根目录: {source_root}")
        print(f"{'=' * 80}")

        for annotator_idx in range(num_annotators):
            print(f"\n{'=' * 80}")
            print(f"处理标注者 {annotator_idx + 1}/{num_annotators}")
            print(f"{'=' * 80}")

            annotations_for_this_annotator = {}
            for video_id, labels_list in annotations_by_annotator.items():
                if annotator_idx < len(labels_list):
                    annotations_for_this_annotator[video_id] = labels_list[
                        annotator_idx
                    ]

            output_root_annotator = source_root / f"annotator_{annotator_idx + 1}"

            _run_batch_comparison(
                cfg=cfg,
                annotations=annotations_for_this_annotator,
                output_root=output_root_annotator,
                threshold_deg=threshold_deg,
                report_title=f"头部姿态估计与人工标注比较 - 论文报告（标注者 {annotator_idx + 1}，{source_label}）",
                smoothed=smoothed,
            )

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
与人工标注比较的运行脚本
使用 config.py 中定义的默认路径
"""
from typing import Optional
from .main import HeadPoseAnalyzer
from .load import load_head_movement_annotations
from .config import (
    get_annotation_file,
    get_fused_dir,
    get_output_dir,
    DEFAULT_PERSON_ID,
    DEFAULT_ENV_JP,
    DEFAULT_START_FRAME,
    DEFAULT_END_FRAME,
    DEFAULT_THRESHOLD,
    ENVIRONMENTS,
)


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


if __name__ == "__main__":
    import sys
    
    # 支持命令行参数
    person_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PERSON_ID
    env_jp = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ENV_JP
    
    run_comparison(person_id=person_id, env_jp=env_jp)

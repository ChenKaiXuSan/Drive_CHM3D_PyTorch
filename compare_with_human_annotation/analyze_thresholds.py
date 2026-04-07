#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
分析标注区间的角度分布
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from head_movement_analysis import (
    HeadPoseAnalyzer,
    load_head_movement_annotations,
)


def analyze_annotation_regions():
    """分析标注区间的角度特征"""
    print("=" * 60)
    print("标注区间角度分析")
    print("=" * 60)
    
    # 加载标注
    annotation_path = Path("/workspace/data/annotation/label/full.json")
    annotations = load_head_movement_annotations(annotation_path)
    
    # 创建分析器
    analyzer = HeadPoseAnalyzer(annotation_dict=annotations)
    
    # 测试数据
    video_id = "01_day_high"
    fused_dir = Path("/workspace/data/head3d_fuse_results/01/昼多い/fused_npz")
    
    # 分析序列
    results = analyzer.analyze_sequence_with_annotations(
        video_id=video_id,
        fused_dir=fused_dir,
        start_frame=380,
        end_frame=520,
    )
    
    # 按标注类型统计角度
    stats_by_label = {}
    
    for frame_idx, comparison in results["comparisons"].items():
        for match_info in comparison["matches"]:
            label = match_info["annotation"].label
            angle_type = match_info["angle_type"]
            angle_value = match_info["angle_value"]
            
            key = f"{label}_{angle_type}"
            if key not in stats_by_label:
                stats_by_label[key] = []
            stats_by_label[key].append(angle_value)
    
    # 显示统计
    print("\n角度统计 (按标注类型):\n")
    for key, values in sorted(stats_by_label.items()):
        label, angle_type = key.rsplit("_", 1)
        print(f"{label} ({angle_type}):")
        print(f"  样本数: {len(values)}")
        print(f"  平均值: {np.mean(values):7.2f}°")
        print(f"  中位数: {np.median(values):7.2f}°")
        print(f"  标准差: {np.std(values):7.2f}°")
        print(f"  最小值: {np.min(values):7.2f}°")
        print(f"  最大值: {np.max(values):7.2f}°")
        print(f"  范围:   {np.min(values):7.2f}° ~ {np.max(values):7.2f}°")
        print()
    
    # 测试不同阈值的匹配率
    print("\n不同阈值下的匹配率:")
    print("-" * 60)
    
    thresholds = [5, 10, 15, 20, 25, 30]
    for threshold in thresholds:
        total_matches = 0
        successful_matches = 0
        
        for comparison in results["comparisons"].values():
            for match_info in comparison["matches"]:
                angle_value = match_info["angle_value"]
                expected_direction = match_info["expected_direction"]
                
                total_matches += 1
                
                # 使用当前阈值判断
                is_match = False
                if expected_direction == "positive":
                    is_match = angle_value > threshold
                elif expected_direction == "negative":
                    is_match = angle_value < -threshold
                
                if is_match:
                    successful_matches += 1
        
        if total_matches > 0:
            match_rate = (successful_matches / total_matches) * 100
            print(f"阈值 {threshold:2d}°: {successful_matches:3d}/{total_matches:3d} = {match_rate:5.1f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    analyze_annotation_regions()

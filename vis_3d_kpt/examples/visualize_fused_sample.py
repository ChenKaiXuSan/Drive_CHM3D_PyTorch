#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
示例：可视化融合3D关键点数据

展示如何加载和可视化 head3d_fuse_results 中的融合3D关键点。
支持：
  - 完整骨架可视化（单帧）
  - 部分骨架可视化（头部、左手、右手）
  - 交互式 3D 视图
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt

from vis_3d_kpt.metadata.mhr70_drive import (
    get_head_hand_indices_mapping,
    pose_info as mhr70_pose_info,
)
from vis_3d_kpt.visualization.scene_visualizer import SceneVisualizer


def load_fused_keypoints(npy_path: Path) -> dict:
    """
    加载融合3D关键点文件。
    
    Args:
        npy_path: 指向 frame_*.npy 的路径
        
    Returns:
        dict 包含：
            - frame_idx: 帧号
            - fused_keypoints_3d: (70, 3) array，融合后的3D关键点
            - fused_mask: 掩码（可能为None）
            - valid_views: 有效视图数
            - npz_paths: 原始NPZ路径信息
    """
    data = np.load(npy_path, allow_pickle=True)
    
    # 处理 object scalar
    if data.dtype == object and data.shape == ():
        data = data.item()
    
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")
    
    return data


def visualize_part_comparison(
    scene_visualizer: SceneVisualizer,
    keypoints_3d: np.ndarray,
    title_prefix: str = "Frame",
    selected_parts: List[str] = None,
):
    """
    在同一坐标系中显示 head、left_hand、right_hand、left_arm、right_arm 的3D骨架。
    
    Args:
        keypoints_3d: (70, 3) 的完整3D关键点
        title_prefix: 标题前缀
        selected_parts: 需要显示的part列表，None表示全部
    """
    mapping = get_head_hand_indices_mapping()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    head_indices = mapping['head']['indices']
    left_hand_indices = mapping['left_hand']['indices']
    right_hand_indices = mapping['right_hand']['indices']
    left_arm_indices = [5, 7, 62]   # left_shoulder, left_elbow, left_wrist
    right_arm_indices = [6, 8, 41]  # right_shoulder, right_elbow, right_wrist

    part_style = {
        'head': dict(indices=head_indices, point_color='red', line_color='red', label=f"Head ({len(head_indices)} pts)"),
        'left_hand': dict(indices=left_hand_indices, point_color='blue', line_color='blue', label=f"Left Hand ({len(left_hand_indices)} pts)"),
        'right_hand': dict(indices=right_hand_indices, point_color='green', line_color='green', label=f"Right Hand ({len(right_hand_indices)} pts)"),
        'left_arm': dict(indices=left_arm_indices, point_color='orange', line_color='orange', label=f"Left Arm ({len(left_arm_indices)} pts)"),
        'right_arm': dict(indices=right_arm_indices, point_color='purple', line_color='purple', label=f"Right Arm ({len(right_arm_indices)} pts)"),
    }

    if selected_parts is None:
        selected_parts = list(part_style.keys())

    edges, _ = scene_visualizer.get_skeleton_edges_with_colors()

    for part_name in selected_parts:
        if part_name not in part_style:
            continue
        style = part_style[part_name]
        indices = style['indices']
        part_kpts = keypoints_3d[indices]
        ax.scatter(
            part_kpts[:, 0],
            part_kpts[:, 1],
            part_kpts[:, 2],
            c=style['point_color'],
            s=50,
            alpha=0.95,
            label=style['label'],
        )

        index_set = set(indices)
        scene_visualizer.draw_skeleton_edges(
            ax=ax,
            keypoints_3d=keypoints_3d,
            edges=edges,
            colors=[style['line_color']] * len(edges),
            part_index_set=index_set,
            linewidth=2.4,
            alpha=0.9,
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title_prefix} - Parts in One Coordinate System ({', '.join(selected_parts)})")
    ax.legend(loc='upper right')
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='融合3D关键点可视化')
    parser.add_argument(
        '--part-draw-option',
        type=str,
        default='head_arm_hand',
        choices=['head', 'head_arm_hand'],
        help='part统一图绘制开关: head=仅头部, head_arm_hand=头+手臂+手',
    )
    args = parser.parse_args()

    part_option_map = {
        'head': ['head'],
        'head_arm_hand': ['head', 'left_arm', 'right_arm', 'left_hand', 'right_hand'],
    }
    selected_parts = part_option_map[args.part_draw_option]

    scene_visualizer = SceneVisualizer(line_width=2, radius=5)
    scene_visualizer.set_pose_meta(mhr70_pose_info)

    # 样本文件路径
    sample_file = Path('/workspace/data/head3d_fuse_results/01/夜多い/smoothed_fused_npz/frame_000620_fused.npy')
    
    print(f"\n{'='*80}")
    print(f"加载融合3D关键点数据")
    print(f"{'='*80}")
    print(f"文件路径: {sample_file}")
    
    # 加载数据
    data = load_fused_keypoints(sample_file)
    print(f"\n✅ 数据加载成功")
    print(f"  Frame Index: {data['frame_idx']}")
    print(f"  3D Keypoints Shape: {data['fused_keypoints_3d'].shape}")
    print(f"  Valid Views: {data['valid_views']}")
    
    keypoints_3d = data['fused_keypoints_3d']
    frame_idx = data['frame_idx']
    
    # 1. 完整骨架可视化
    print(f"\n{'='*80}")
    print(f"1. 完整骨架可视化（所有70个关键点）")
    print(f"{'='*80}")
    fig1, ax1 = scene_visualizer.visualize_3d_skeleton(
        keypoints_3d,
        title=f"Complete Skeleton - Frame {frame_idx} (70 keypoints)"
    )
    plt.savefig('/workspace/code/logs/vis_3d_kpt/complete_skeleton.png', dpi=150, bbox_inches='tight')
    print(f"✅ 已保存: /workspace/code/logs/vis_3d_kpt/complete_skeleton.png")
    plt.close(fig1)
    
    # 2. 部分骨架同一坐标系显示（可选保存）
    print(f"\n{'='*80}")
    print(f"2. 部分骨架同一坐标系（绘制开关: {args.part_draw_option}）")
    print(f"{'='*80}")
    fig2 = visualize_part_comparison(
        scene_visualizer,
        keypoints_3d,
        title_prefix=f"Frame {frame_idx}",
        selected_parts=selected_parts,
    )
    part_save_path = '/workspace/code/logs/vis_3d_kpt/part_comparison.png'
    plt.savefig(part_save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存: {part_save_path}")
    plt.close(fig2)
    
    print(f"\n{'='*80}")
    print(f"✅ 可视化已完成（仅保存统一图，不分开保存part）")
    print(f"{'='*80}\n")
    
    # 统计信息
    print(f"\n【3D关键点统计】")
    print(f"  X 范围: [{keypoints_3d[:, 0].min():.3f}, {keypoints_3d[:, 0].max():.3f}]")
    print(f"  Y 范围: [{keypoints_3d[:, 1].min():.3f}, {keypoints_3d[:, 1].max():.3f}]")
    print(f"  Z 范围: [{keypoints_3d[:, 2].min():.3f}, {keypoints_3d[:, 2].max():.3f}]")
    
    print("\n✅ 已按配置完成单帧可视化（未执行多帧比较）")


if __name__ == "__main__":
    main()

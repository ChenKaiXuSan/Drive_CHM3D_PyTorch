"""
示例：如何按部分可视化头部和手部关键点

这个示例展示了如何使用 SkeletonVisualizer 的 part_indices 参数
来只可视化头部、左手、或右手的关键点。
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from vis_3d_kpt.metadata.mhr70_drive import (
    get_head_hand_indices_mapping,
    get_subset_keypoints,
)

# 注：SkeletonVisualizer 导入可能需要 matplotlib 等依赖
# from vis_3d_kpt.visualization.skeleton_visualizer import SkeletonVisualizer


def example_2d_visualization():
    """2D 可视化示例"""
    print("\n" + "="*80)
    print("2D 部分可视化示例")
    print("="*80)
    
    # 需要 SkeletonVisualizer（需要 matplotlib 等依赖）
    # from vis_3d_kpt.visualization.skeleton_visualizer import SkeletonVisualizer
    # visualizer = SkeletonVisualizer(...)
    pass


def example_3d_visualization():
    """3D 可视化示例"""
    print("\n" + "="*80)
    print("3D 部分可视化示例")
    print("="*80)
    
    # 需要 SkeletonVisualizer（需要 matplotlib 等依赖）
    # from vis_3d_kpt.visualization.skeleton_visualizer import SkeletonVisualizer
    # visualizer = SkeletonVisualizer(...)
    pass


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "="*80)
    print("部分可视化 - 使用指南")
    print("="*80)
    
    print("\n【基础用法】")
    print("""
1. 导入必要模块：
   from vis_3d_kpt.metadata.mhr70_drive import get_subset_keypoints
   from vis_3d_kpt.visualization.skeleton_visualizer import SkeletonVisualizer

2. 初始化可视化器：
   visualizer = SkeletonVisualizer(...)

3. 获取部分索引：
   head_info = get_subset_keypoints('head')
   head_indices = head_info['indices']  # [0, 1, 2, 3, 4, 5, 6, 69]

4. 可视化指定部分：
   
   # 2D 可视化
   image = visualizer.draw_skeleton(
       image,
       keypoints,
       part_indices=head_indices,  # 只绘制头部
   )
   
   # 3D 可视化
   fig = visualizer.draw_skeleton_3d(
       ax,
       points_3d,
       part_indices=head_indices,  # 只绘制头部
   )
    """)
    
    print("\n【可用部分】")
    mapping = get_head_hand_indices_mapping()
    for part_name, info in mapping.items():
        print(f"\n  {part_name.upper()}:")
        print(f"    - 索引: {info['indices']}")
        print(f"    - 数量: {info['count']} 个关键点")
        print(f"    - 描述: {info['description']}")
    
    print("\n【获取部分信息的方式】")
    print("""
  # 方式 1：通过 get_subset_keypoints 获取单个部分
  head_info = get_subset_keypoints('head')
  head_indices = head_info['indices']
  
  # 方式 2：通过 get_head_hand_indices_mapping 获取所有部分
  mapping = get_head_hand_indices_mapping()
  head_indices = mapping['head']['indices']
  left_hand_indices = mapping['left_hand']['indices']
  right_hand_indices = mapping['right_hand']['indices']
  
  # 方式 3：直接导入
  from vis_3d_kpt.metadata.mhr70_drive import (
      head_keypoint_indices,
      left_hand_keypoint_indices,
      right_hand_keypoint_indices,
  )
    """)
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # 打印使用指南
    print_usage_guide()
    
    # 注：以下代码需要 matplotlib、opencv 等可视化库，
    # 如果环境准备好了，可以取消注释运行
    
    # try:
    #     example_2d_visualization()
    # except Exception as e:
    #     print(f"\n2D 示例出错: {e}")
    # 
    # try:
    #     example_3d_visualization()
    # except Exception as e:
    #     print(f"\n3D 示例出错: {e}")

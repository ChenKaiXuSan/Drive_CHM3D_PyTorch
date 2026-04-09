#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/head3D_fuse/fuse/compare_fused.py
Project: /workspace/code/head3D_fuse/fuse
Created Date: Thursday February 5th 2026
Author: Kaixu Chen
-----
Comment:
比较融合后的3D关键点与各个单视角3D关键点的质量评估

Have a good code time :)
-----
Last Modified: Thursday February 5th 2026 9:07:43 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


class FusedViewComparator:
    """比较融合后的3D关键点与各个单视角3D关键点的评估器
    
    评估指标包括：
    1. 与各视角的平均欧氏距离
    2. 融合带来的时间稳定性提升（jitter reduction）
    3. 各关键点在不同视角的一致性分析
    4. 融合结果的置信度评估
    """
    
    def __init__(
        self,
        fused_keypoints: np.ndarray,
        view_keypoints: Dict[str, np.ndarray],
    ):
        """初始化比较器
        
        Args:
            fused_keypoints: 融合后的关键点，形状 (T, N, 3)
            view_keypoints: 各视角的关键点字典 {"front": (T, N, 3), "left": (T, N, 3), "right": (T, N, 3)}
        """
        self.fused = fused_keypoints
        self.views = view_keypoints
        self.view_names = sorted(view_keypoints.keys())
        
        # 验证数据形状
        self._validate_data()
        
        self.T, self.N, _ = self.fused.shape
        logger.info(f"Initialized FusedViewComparator: {self.T} frames, {self.N} keypoints, {len(self.view_names)} views")
    
    def _validate_data(self):
        """验证输入数据的形状和一致性"""
        if self.fused.ndim != 3:
            raise ValueError(f"fused_keypoints should be 3D array (T, N, 3), got shape {self.fused.shape}")
        
        T, N, _ = self.fused.shape
        for view_name, kpts in self.views.items():
            if kpts.shape != self.fused.shape:
                raise ValueError(
                    f"View '{view_name}' keypoints shape {kpts.shape} does not match fused shape {self.fused.shape}"
                )
    
    def compute_euclidean_distances(
        self, 
        keypoint_indices: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """计算融合关键点与各视角关键点的欧氏距离
        
        Args:
            keypoint_indices: 要计算的关键点索引列表，None表示所有关键点
            
        Returns:
            字典 {view_name: distances (T, N)} 每帧每个关键点的距离
        """
        if keypoint_indices is None:
            keypoint_indices = list(range(self.N))
        
        distances = {}
        for view_name in self.view_names:
            view_kpts = self.views[view_name]
            # 计算欧氏距离: sqrt(sum((fused - view)^2))
            dist = np.linalg.norm(
                self.fused[:, keypoint_indices, :] - view_kpts[:, keypoint_indices, :],
                axis=2
            )
            distances[view_name] = dist
            
        return distances
    
    def compute_temporal_jitter(
        self, 
        keypoints: np.ndarray,
        keypoint_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """计算关键点序列的时间抖动（加速度）
        
        Args:
            keypoints: 关键点序列 (T, N, 3)
            keypoint_indices: 要计算的关键点索引
            
        Returns:
            抖动值 (T-2, N)，每个关键点的加速度
        """
        if keypoint_indices is None:
            keypoint_indices = list(range(keypoints.shape[1]))
        
        kpts = keypoints[:, keypoint_indices, :]
        
        # 计算加速度：二阶差分
        velocity = np.diff(kpts, axis=0)  # (T-1, N, 3)
        acceleration = np.diff(velocity, axis=0)  # (T-2, N, 3)
        jitter = np.linalg.norm(acceleration, axis=2)  # (T-2, N)
        
        return jitter
    
    def compute_view_consistency(
        self,
        keypoint_indices: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """计算各视角之间的一致性（两两视角的距离）
        
        Args:
            keypoint_indices: 要计算的关键点索引
            
        Returns:
            字典 {(view1, view2): distances (T, N)}
        """
        if keypoint_indices is None:
            keypoint_indices = list(range(self.N))
        
        consistency = {}
        for i, view1 in enumerate(self.view_names):
            for view2 in self.view_names[i+1:]:
                dist = np.linalg.norm(
                    self.views[view1][:, keypoint_indices, :] - 
                    self.views[view2][:, keypoint_indices, :],
                    axis=2
                )
                consistency[f"{view1}_vs_{view2}"] = dist
        
        return consistency
    
    def compute_metrics(
        self,
        keypoint_indices: Optional[List[int]] = None
    ) -> Dict[str, Union[float, Dict]]:
        """计算所有评估指标
        
        Args:
            keypoint_indices: 要评估的关键点索引
            
        Returns:
            包含所有指标的字典
        """
        if keypoint_indices is None:
            keypoint_indices = list(range(self.N))
        
        metrics = {}
        
        # 1. 与各视角的平均距离
        distances = self.compute_euclidean_distances(keypoint_indices)
        metrics["mean_distance_to_views"] = {
            view: float(np.nanmean(dist)) for view, dist in distances.items()
        }
        metrics["std_distance_to_views"] = {
            view: float(np.nanstd(dist)) for view, dist in distances.items()
        }
        
        # 2. 计算融合结果是否在各视角的"中心"
        all_view_kpts = np.stack([self.views[v][:, keypoint_indices, :] for v in self.view_names], axis=0)
        view_centroid = np.nanmean(all_view_kpts, axis=0)  # (T, N, 3)
        centroid_distance = np.linalg.norm(
            self.fused[:, keypoint_indices, :] - view_centroid, axis=2
        )
        metrics["mean_distance_to_centroid"] = float(np.nanmean(centroid_distance))
        metrics["std_distance_to_centroid"] = float(np.nanstd(centroid_distance))
        
        # 3. 时间稳定性：比较融合结果与各视角的jitter
        fused_jitter = self.compute_temporal_jitter(self.fused, keypoint_indices)
        metrics["fused_jitter"] = {
            "mean": float(np.nanmean(fused_jitter)),
            "std": float(np.nanstd(fused_jitter)),
        }
        
        view_jitters = {}
        for view_name in self.view_names:
            view_jitter = self.compute_temporal_jitter(self.views[view_name], keypoint_indices)
            view_jitters[view_name] = {
                "mean": float(np.nanmean(view_jitter)),
                "std": float(np.nanstd(view_jitter)),
            }
            # 计算jitter reduction
            reduction = (np.nanmean(view_jitter) - np.nanmean(fused_jitter)) / np.nanmean(view_jitter) * 100
            view_jitters[view_name]["jitter_reduction_vs_fused"] = float(reduction)
        
        metrics["view_jitters"] = view_jitters
        
        # 4. 视角一致性
        consistency = self.compute_view_consistency(keypoint_indices)
        metrics["view_consistency"] = {
            pair: {
                "mean": float(np.nanmean(dist)),
                "std": float(np.nanstd(dist)),
            }
            for pair, dist in consistency.items()
        }
        
        # 5. 每个关键点的详细统计
        per_keypoint_metrics = {}
        for idx in keypoint_indices:
            kp_metrics = {}
            # 与各视角的平均距离
            for view_name in self.view_names:
                dist = np.linalg.norm(
                    self.fused[:, idx, :] - self.views[view_name][:, idx, :],
                    axis=1
                )
                kp_metrics[f"mean_dist_to_{view_name}"] = float(np.nanmean(dist))
            
            per_keypoint_metrics[f"keypoint_{idx}"] = kp_metrics
        
        metrics["per_keypoint_metrics"] = per_keypoint_metrics
        
        return metrics
    
    def generate_report(
        self,
        save_path: Optional[Path] = None,
        keypoint_indices: Optional[List[int]] = None
    ) -> str:
        """生成详细的文本报告
        
        Args:
            save_path: 保存路径，None则不保存
            keypoint_indices: 要评估的关键点索引
            
        Returns:
            报告文本
        """
        metrics = self.compute_metrics(keypoint_indices)
        
        lines = []
        lines.append("=" * 80)
        lines.append("融合关键点与单视角关键点对比评估报告")
        lines.append("=" * 80)
        lines.append(f"数据维度: {self.T} 帧, {self.N} 关键点, {len(self.view_names)} 视角")
        lines.append(f"评估的关键点索引: {keypoint_indices if keypoint_indices else 'All'}")
        lines.append("")
        
        # 1. 与各视角的距离
        lines.append("-" * 80)
        lines.append("1. 融合结果与各视角的平均距离（越小越接近该视角）")
        lines.append("-" * 80)
        for view in self.view_names:
            mean_dist = metrics["mean_distance_to_views"][view]
            std_dist = metrics["std_distance_to_views"][view]
            lines.append(f"  {view:>10s}: {mean_dist:8.4f} ± {std_dist:8.4f}")
        lines.append("")
        
        # 2. 与质心的距离
        lines.append("-" * 80)
        lines.append("2. 融合结果与视角质心的距离（越小说明融合越居中）")
        lines.append("-" * 80)
        lines.append(f"  Mean: {metrics['mean_distance_to_centroid']:.4f}")
        lines.append(f"  Std:  {metrics['std_distance_to_centroid']:.4f}")
        lines.append("")
        
        # 3. 时间稳定性对比
        lines.append("-" * 80)
        lines.append("3. 时间稳定性对比（Jitter = 加速度，越小越稳定）")
        lines.append("-" * 80)
        lines.append(f"  融合结果 Jitter: {metrics['fused_jitter']['mean']:.6f} ± {metrics['fused_jitter']['std']:.6f}")
        lines.append("")
        lines.append("  各视角 Jitter 及相对融合结果的改善:")
        for view in self.view_names:
            jitter_data = metrics["view_jitters"][view]
            lines.append(f"    {view:>10s}: {jitter_data['mean']:.6f} ± {jitter_data['std']:.6f} "
                        f"(改善: {jitter_data['jitter_reduction_vs_fused']:+.2f}%)")
        lines.append("")
        
        # 4. 视角一致性
        lines.append("-" * 80)
        lines.append("4. 视角间一致性（两两视角的差异，越小说明原始数据越一致）")
        lines.append("-" * 80)
        for pair, stats in metrics["view_consistency"].items():
            lines.append(f"  {pair:>20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        lines.append("")
        
        # 5. 关键点级别分析（只展示前10个）
        lines.append("-" * 80)
        lines.append("5. 各关键点与视角的平均距离（前10个关键点）")
        lines.append("-" * 80)
        kp_items = list(metrics["per_keypoint_metrics"].items())[:10]
        for kp_name, kp_data in kp_items:
            lines.append(f"  {kp_name}:")
            for key, val in kp_data.items():
                lines.append(f"    {key}: {val:.4f}")
        lines.append("")
        
        lines.append("=" * 80)
        lines.append("报告结束")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report
    
    def plot_comparison(
        self,
        save_path: Optional[Path] = None,
        keypoint_indices: Optional[List[int]] = None,
        figsize: tuple = (16, 12),
    ):
        """绘制对比可视化图表
        
        Args:
            save_path: 保存路径
            keypoint_indices: 要可视化的关键点索引
            figsize: 图表尺寸
        """
        if keypoint_indices is None:
            keypoint_indices = list(range(min(7, self.N)))  # 默认前7个
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle("Fused vs Single-view Keypoints Analysis", fontsize=16, fontweight="bold")
        
        # 1. 与各视角的平均距离 - 柱状图
        ax = axes[0, 0]
        distances = self.compute_euclidean_distances(keypoint_indices)
        mean_dists = [np.nanmean(distances[v]) for v in self.view_names]
        std_dists = [np.nanstd(distances[v]) for v in self.view_names]
        ax.bar(self.view_names, mean_dists, yerr=std_dists, capsize=5, alpha=0.7)
        ax.set_title("Mean Distance from Fused to Each View")
        ax.set_ylabel("Euclidean Distance")
        ax.grid(True, alpha=0.3)
        
        # 2. 时间稳定性对比 - Jitter
        ax = axes[0, 1]
        fused_jitter = self.compute_temporal_jitter(self.fused, keypoint_indices)
        jitter_data = [np.nanmean(fused_jitter)]
        jitter_labels = ["Fused"]
        for view in self.view_names:
            view_jitter = self.compute_temporal_jitter(self.views[view], keypoint_indices)
            jitter_data.append(np.nanmean(view_jitter))
            jitter_labels.append(view.capitalize())
        ax.bar(jitter_labels, jitter_data, alpha=0.7, color=['green'] + ['blue'] * len(self.view_names))
        ax.set_title("Temporal Stability Comparison (Jitter)")
        ax.set_ylabel("Mean Acceleration")
        ax.grid(True, alpha=0.3)
        
        # 3. 各关键点与各视角的距离热图
        ax = axes[1, 0]
        distances_matrix = []
        for view in self.view_names:
            view_dists = []
            for idx in keypoint_indices:
                dist = np.linalg.norm(
                    self.fused[:, idx, :] - self.views[view][:, idx, :],
                    axis=1
                )
                view_dists.append(np.nanmean(dist))
            distances_matrix.append(view_dists)
        
        sns.heatmap(
            distances_matrix,
            ax=ax,
            xticklabels=[f"KP{i}" for i in keypoint_indices],
            yticklabels=[v.capitalize() for v in self.view_names],
            annot=True,
            fmt=".3f",
            cmap="YlOrRd"
        )
        ax.set_title("Mean Distance per Keypoint and View")
        
        # 4. 选择特定关键点的时序轨迹对比（以第一个关键点为例）
        ax = axes[1, 1]
        kp_idx = keypoint_indices[0]
        ax.plot(self.fused[:, kp_idx, 0], label="Fused", linewidth=2, alpha=0.8)
        for view in self.view_names:
            ax.plot(self.views[view][:, kp_idx, 0], label=view.capitalize(), alpha=0.6)
        ax.set_title(f"Keypoint {kp_idx} X-coordinate Over Time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("X Coordinate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 视角一致性矩阵
        ax = axes[2, 0]
        consistency = self.compute_view_consistency(keypoint_indices)
        n_views = len(self.view_names)
        consistency_matrix = np.zeros((n_views, n_views))
        for i, view1 in enumerate(self.view_names):
            for j, view2 in enumerate(self.view_names):
                if i == j:
                    consistency_matrix[i, j] = 0
                elif i < j:
                    key = f"{view1}_vs_{view2}"
                    consistency_matrix[i, j] = np.nanmean(consistency[key])
                    consistency_matrix[j, i] = consistency_matrix[i, j]
        
        sns.heatmap(
            consistency_matrix,
            ax=ax,
            xticklabels=[v.capitalize() for v in self.view_names],
            yticklabels=[v.capitalize() for v in self.view_names],
            annot=True,
            fmt=".3f",
            cmap="coolwarm"
        )
        ax.set_title("Inter-view Consistency (Pairwise Distance)")
        
        # 6. 每帧的平均距离时序图
        ax = axes[2, 1]
        for view in self.view_names:
            distances_per_frame = np.nanmean(distances[view], axis=1)
            ax.plot(distances_per_frame, label=view.capitalize(), alpha=0.7)
        ax.set_title("Per-frame Mean Distance: Fused vs Views")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean Distance")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
            plt.close()
        else:
            plt.show()
    
    def export_metrics_json(
        self,
        save_path: Path,
        keypoint_indices: Optional[List[int]] = None
    ):
        """导出指标到JSON文件
        
        Args:
            save_path: 保存路径
            keypoint_indices: 要评估的关键点索引
        """
        metrics = self.compute_metrics(keypoint_indices)
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Metrics exported to {save_path}")

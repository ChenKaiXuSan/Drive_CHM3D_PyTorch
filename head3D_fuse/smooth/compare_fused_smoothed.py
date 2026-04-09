#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
比较融合后的3D关键点和平滑后的3D关键点效果

提供多种评估指标和可视化方法来分析时间平滑的效果。

Author: Kaixu Chen
Created: February 2026
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean

logger = logging.getLogger(__name__)


class KeypointsComparator:
    """比较融合关键点和平滑关键点的工具类"""

    def __init__(self, fused_kpts: np.ndarray, smoothed_kpts: np.ndarray):
        """
        初始化比较器

        Args:
            fused_kpts: 融合后的关键点 (T, N, 3)
            smoothed_kpts: 平滑后的关键点 (T, N, 3)
        """
        self.fused_kpts = np.asarray(fused_kpts, dtype=np.float32)
        self.smoothed_kpts = np.asarray(smoothed_kpts, dtype=np.float32)

        if self.fused_kpts.shape != self.smoothed_kpts.shape:
            raise ValueError(
                f"Shape mismatch: fused {self.fused_kpts.shape} "
                f"vs smoothed {self.smoothed_kpts.shape}"
            )

        self.T, self.N, _ = self.fused_kpts.shape
        logger.info(f"Initialized with {self.T} frames, {self.N} keypoints")

    def compute_point_wise_difference(self) -> np.ndarray:
        """
        计算逐点差异（L2距离）

        Returns:
            (T, N) 数组，表示每个关键点在每帧的差异
        """
        diff = self.smoothed_kpts - self.fused_kpts
        point_wise_dist = np.linalg.norm(diff, axis=-1)
        return point_wise_dist

    def compute_frame_wise_difference(self) -> np.ndarray:
        """
        计算逐帧差异（所有关键点的平均差异）

        Returns:
            (T,) 数组，表示每帧的平均差异
        """
        point_wise = self.compute_point_wise_difference()
        frame_wise = np.mean(point_wise, axis=1)
        return frame_wise

    def compute_smoothness(self, kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算时间平滑度（速度和加速度的幅度）

        Args:
            kpts: (T, N, 3) 关键点数组

        Returns:
            velocity_magnitude: (T-1, N) 速度幅度
            acceleration_magnitude: (T-2, N) 加速度幅度
        """
        # 速度 (一阶导数)
        velocity = kpts[1:] - kpts[:-1]
        velocity_mag = np.linalg.norm(velocity, axis=-1)

        # 加速度 (二阶导数)
        acceleration = velocity[1:] - velocity[:-1]
        acceleration_mag = np.linalg.norm(acceleration, axis=-1)

        return velocity_mag, acceleration_mag

    def compute_jitter(self, kpts: np.ndarray) -> float:
        """
        计算抖动程度（加速度的标准差）

        Args:
            kpts: (T, N, 3) 关键点数组

        Returns:
            抖动分数（越小越平滑）
        """
        _, accel_mag = self.compute_smoothness(kpts)
        jitter_score = np.mean(np.std(accel_mag, axis=0))
        return float(jitter_score)

    def compute_metrics(self, keypoint_indices: Optional[List[int]] = None) -> Dict[str, float]:
        """
        计算所有评估指标

        Args:
            keypoint_indices: 要计算的关键点索引（None表示所有关键点）

        Returns:
            包含各种指标的字典
        """
        point_wise = self.compute_point_wise_difference()
        frame_wise = self.compute_frame_wise_difference()

        # 如果指定了关键点索引，只选择这些关键点
        if keypoint_indices is not None:
            point_wise = point_wise[:, keypoint_indices]
            frame_wise = np.mean(point_wise, axis=1)

        # 平滑度指标
        fused_kpts_subset = self.fused_kpts if keypoint_indices is None else self.fused_kpts[:, keypoint_indices, :]
        smoothed_kpts_subset = self.smoothed_kpts if keypoint_indices is None else self.smoothed_kpts[:, keypoint_indices, :]
        
        fused_vel, fused_accel = self.compute_smoothness(fused_kpts_subset)
        smooth_vel, smooth_accel = self.compute_smoothness(smoothed_kpts_subset)

        metrics = {
            # 差异指标
            "mean_difference": float(np.mean(point_wise)),
            "max_difference": float(np.max(point_wise)),
            "std_difference": float(np.std(point_wise)),
            "median_difference": float(np.median(point_wise)),
            # 速度指标
            "fused_mean_velocity": float(np.mean(fused_vel)),
            "smoothed_mean_velocity": float(np.mean(smooth_vel)),
            "velocity_reduction": float(
                (np.mean(fused_vel) - np.mean(smooth_vel)) / np.mean(fused_vel) * 100
                if np.mean(fused_vel) > 0 else 0
            ),
            # 加速度指标（抖动）
            "fused_mean_acceleration": float(np.mean(fused_accel)),
            "smoothed_mean_acceleration": float(np.mean(smooth_accel)),
            "acceleration_reduction": float(
                (np.mean(fused_accel) - np.mean(smooth_accel))
                / np.mean(fused_accel)
                * 100
                if np.mean(fused_accel) > 0 else 0
            ),
            # 抖动分数
            "fused_jitter": self.compute_jitter(fused_kpts_subset),
            "smoothed_jitter": self.compute_jitter(smoothed_kpts_subset),
            "jitter_reduction": float(
                (self.compute_jitter(fused_kpts_subset) - self.compute_jitter(smoothed_kpts_subset))
                / self.compute_jitter(fused_kpts_subset)
                * 100
                if self.compute_jitter(fused_kpts_subset) > 0 else 0
            ),
        }

        return metrics

    def plot_comparison(
        self, save_path: Optional[Path] = None, keypoint_indices: Optional[List[int]] = None
    ):
        """
        Plot trajectory comparison.

        Args:
            save_path: Output path for the figure
            keypoint_indices: Keypoint indices to plot (default: first 3)
        """
        if keypoint_indices is None:
            keypoint_indices = list(range(min(3, self.N)))

        num_kpts = len(keypoint_indices)
        fig, axes = plt.subplots(num_kpts, 3, figsize=(15, 4 * num_kpts))
        if num_kpts == 1:
            axes = axes.reshape(1, -1)

        t = np.arange(self.T)

        for i, kpt_idx in enumerate(keypoint_indices):
            for dim, dim_name in enumerate(["X", "Y", "Z"]):
                ax = axes[i, dim]

                # Plot fused and smoothed trajectories.
                ax.plot(
                    t,
                    self.fused_kpts[:, kpt_idx, dim],
                    "o-",
                    alpha=0.5,
                    label="Fused",
                    markersize=3,
                )
                ax.plot(
                    t,
                    self.smoothed_kpts[:, kpt_idx, dim],
                    "-",
                    label="Smoothed",
                    linewidth=2,
                )

                ax.set_title(f"Keypoint {kpt_idx} - {dim_name} axis")
                ax.set_xlabel("Frame")
                ax.set_ylabel(f"{dim_name} coordinate")
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_metrics(self, save_path: Optional[Path] = None, keypoint_indices: Optional[List[int]] = None):
        """
        Plot metric overview figures.

        Args:
            save_path: Output path for the figure
            keypoint_indices: Keypoint indices to show (None means all keypoints)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Frame-wise difference
        ax = axes[0, 0]
        frame_diff = self.compute_frame_wise_difference()
        ax.plot(frame_diff, "-o", markersize=3)
        ax.set_title("Frame-wise Difference (Mean per frame)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Mean L2 Distance")
        ax.grid(True, alpha=0.3)

        # 2. Velocity comparison
        ax = axes[0, 1]
        fused_vel, _ = self.compute_smoothness(self.fused_kpts)
        smooth_vel, _ = self.compute_smoothness(self.smoothed_kpts)

        fused_vel_mean = np.mean(fused_vel, axis=1)
        smooth_vel_mean = np.mean(smooth_vel, axis=1)

        ax.plot(fused_vel_mean, label="Fused", alpha=0.7)
        ax.plot(smooth_vel_mean, label="Smoothed", alpha=0.7)
        ax.set_title("Velocity Magnitude (Mean per frame)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Velocity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Acceleration comparison
        ax = axes[1, 0]
        _, fused_accel = self.compute_smoothness(self.fused_kpts)
        _, smooth_accel = self.compute_smoothness(self.smoothed_kpts)

        fused_accel_mean = np.mean(fused_accel, axis=1)
        smooth_accel_mean = np.mean(smooth_accel, axis=1)

        ax.plot(fused_accel_mean, label="Fused", alpha=0.7)
        ax.plot(smooth_accel_mean, label="Smoothed", alpha=0.7)
        ax.set_title("Acceleration Magnitude (Mean per frame)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Acceleration")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Keypoint-wise difference distribution (optional index filtering)
        ax = axes[1, 1]
        point_wise = self.compute_point_wise_difference()
        
        if keypoint_indices is not None:
            # Show only selected keypoints.
            point_wise_filtered = point_wise[:, keypoint_indices]
            kpt_diff_mean = np.mean(point_wise_filtered, axis=0)
            x_labels = [str(idx) for idx in keypoint_indices]
            x_positions = range(len(keypoint_indices))
            title_suffix = f" (Keypoints: {keypoint_indices})"
        else:
            # Show differences for all keypoints.
            kpt_diff_mean = np.mean(point_wise, axis=0)
            x_positions = range(len(kpt_diff_mean))
            x_labels = [str(i) for i in range(self.N)]
            title_suffix = " (All)"

        ax.bar(x_positions, kpt_diff_mean, color='steelblue', alpha=0.7)
        ax.set_title(f"Mean Difference per Keypoint{title_suffix}")
        ax.set_xlabel("Keypoint Index")
        ax.set_ylabel("Mean L2 Distance")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45 if len(x_labels) > 10 else 0)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved metrics plot to {save_path}")
        else:
            plt.show()

        plt.close()

    def generate_report(self, save_path: Optional[Path] = None, keypoint_indices: Optional[List[int]] = None) -> str:
        """
        生成文本报告

        Args:
            save_path: 保存路径
            keypoint_indices: 要计算的关键点索引（None表示所有关键点）

        Returns:
            报告文本
        """
        metrics = self.compute_metrics(keypoint_indices=keypoint_indices)

        report = []
        report.append("=" * 70)
        report.append("融合关键点 vs 平滑关键点 - 对比报告")
        report.append("=" * 70)
        report.append(f"\n数据概览:")
        report.append(f"  帧数: {self.T}")
        report.append(f"  关键点数: {self.N}")
        if keypoint_indices is not None:
            report.append(f"  评估关键点: {keypoint_indices} ({len(keypoint_indices)} 个)")
        report.append(f"  数据形状: {self.fused_kpts.shape}")

        report.append(f"\n" + "-" * 70)
        report.append("差异指标:")
        report.append("-" * 70)
        report.append(f"  平均差异:   {metrics['mean_difference']:.6f}")
        report.append(f"  最大差异:   {metrics['max_difference']:.6f}")
        report.append(f"  标准差:     {metrics['std_difference']:.6f}")
        report.append(f"  中位数:     {metrics['median_difference']:.6f}")

        report.append(f"\n" + "-" * 70)
        report.append("平滑度指标 - 速度:")
        report.append("-" * 70)
        report.append(f"  融合后平均速度:     {metrics['fused_mean_velocity']:.6f}")
        report.append(f"  平滑后平均速度:     {metrics['smoothed_mean_velocity']:.6f}")
        report.append(f"  速度降低:           {metrics['velocity_reduction']:.2f}%")

        report.append(f"\n" + "-" * 70)
        report.append("平滑度指标 - 加速度（抖动）:")
        report.append("-" * 70)
        report.append(f"  融合后平均加速度:   {metrics['fused_mean_acceleration']:.6f}")
        report.append(f"  平滑后平均加速度:   {metrics['smoothed_mean_acceleration']:.6f}")
        report.append(f"  加速度降低:         {metrics['acceleration_reduction']:.2f}%")

        report.append(f"\n" + "-" * 70)
        report.append("抖动分数:")
        report.append("-" * 70)
        report.append(f"  融合后抖动分数:     {metrics['fused_jitter']:.6f}")
        report.append(f"  平滑后抖动分数:     {metrics['smoothed_jitter']:.6f}")
        report.append(f"  抖动降低:           {metrics['jitter_reduction']:.2f}%")

        report.append(f"\n" + "=" * 70)
        report.append("评估结论:")
        report.append("=" * 70)

        # 自动评估
        if metrics["acceleration_reduction"] > 20:
            report.append("  ✓ 优秀: 加速度显著降低，抖动明显减少")
        elif metrics["acceleration_reduction"] > 10:
            report.append("  ✓ 良好: 加速度适度降低，平滑效果明显")
        elif metrics["acceleration_reduction"] > 5:
            report.append("  ○ 一般: 加速度略有降低，平滑效果有限")
        else:
            report.append("  ✗ 较差: 加速度降低不明显，考虑调整参数")

        if metrics["mean_difference"] < 0.01:
            report.append("  ✓ 平滑保真度高: 与原始数据差异很小")
        elif metrics["mean_difference"] < 0.05:
            report.append("  ○ 平滑保真度中等: 与原始数据有一定差异")
        else:
            report.append("  ✗ 平滑保真度低: 与原始数据差异较大，可能过度平滑")

        report.append("=" * 70)

        report_text = "\n".join(report)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(report_text)
            logger.info(f"Saved report to {save_path}")

        return report_text


def load_keypoints_from_npz_dir(npz_dir: Path) -> np.ndarray:
    """
    从npz目录加载关键点序列

    Args:
        npz_dir: npz文件目录

    Returns:
        (T, N, 3) 关键点数组
    """
    npz_files = sorted(npz_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")

    keypoints_list = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        # 尝试不同的键名
        if "fused_keypoints" in data:
            kpts = data["fused_keypoints"]
        elif "pred_keypoints_3d" in data:
            kpts = data["pred_keypoints_3d"]
        elif "keypoints_3d" in data:
            kpts = data["keypoints_3d"]
        else:
            logger.warning(f"Unknown keypoints key in {npz_file}, available: {list(data.keys())}")
            continue

        keypoints_list.append(kpts)

    if not keypoints_list:
        raise ValueError(f"No valid keypoints found in {npz_dir}")

    keypoints_array = np.stack(keypoints_list, axis=0)
    logger.info(f"Loaded {len(keypoints_list)} frames from {npz_dir}")

    return keypoints_array


def compare_fused_and_smoothed(
    fused_dir: Path,
    smoothed_dir: Path,
    output_dir: Path,
    keypoint_indices: Optional[List[int]] = None,
):
    """
    比较融合关键点和平滑关键点的主函数

    Args:
        fused_dir: 融合关键点的npz目录
        smoothed_dir: 平滑关键点的npz目录
        output_dir: 输出目录
        keypoint_indices: 要可视化的关键点索引
    """
    logger.info("=" * 70)
    logger.info("开始比较融合和平滑关键点")
    logger.info("=" * 70)

    # 1. 加载数据
    logger.info(f"加载融合关键点: {fused_dir}")
    fused_kpts = load_keypoints_from_npz_dir(fused_dir)

    logger.info(f"加载平滑关键点: {smoothed_dir}")
    smoothed_kpts = load_keypoints_from_npz_dir(smoothed_dir)

    # 2. 创建比较器
    comparator = KeypointsComparator(fused_kpts, smoothed_kpts)

    # 3. 计算指标
    logger.info("计算评估指标...")
    metrics = comparator.compute_metrics()

    # 4. 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标到JSON
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ 保存指标到 {metrics_path}")

    # 生成并保存报告
    report_path = output_dir / "comparison_report.txt"
    report = comparator.generate_report(save_path=report_path)
    print("\n" + report)

    # 绘制对比图
    logger.info("生成可视化...")
    comparison_plot_path = output_dir / "trajectory_comparison.png"
    comparator.plot_comparison(
        save_path=comparison_plot_path, keypoint_indices=keypoint_indices
    )

    metrics_plot_path = output_dir / "metrics_comparison.png"
    comparator.plot_metrics(save_path=metrics_plot_path)

    logger.info("=" * 70)
    logger.info(f"✓ 比较完成！结果保存到: {output_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    # 示例用法
    import argparse

    parser = argparse.ArgumentParser(description="比较融合和平滑的3D关键点")
    parser.add_argument("--fused_dir", type=str, required=True, help="融合关键点目录")
    parser.add_argument("--smoothed_dir", type=str, required=True, help="平滑关键点目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--keypoints",
        type=int,
        nargs="+",
        default=None,
        help="要可视化的关键点索引（默认前3个）",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    compare_fused_and_smoothed(
        fused_dir=Path(args.fused_dir),
        smoothed_dir=Path(args.smoothed_dir),
        output_dir=Path(args.output_dir),
        keypoint_indices=args.keypoints,
    )

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/head3D_fuse/fuse.py
Project: /workspace/code/head3D_fuse
Created Date: Monday February 2nd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday February 2nd 2026 6:47:37 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import json
import logging
import gc
from pathlib import Path
from typing import Optional, cast

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

# load
from head3D_fuse.load import (
    assemble_view_npz_paths,
    compare_npz_files,
    get_annotation_dict,
    load_npz_output,
)

# save
from head3D_fuse.save import _save_fused_keypoints

# vis
from head3D_fuse.visualization.merge_video import merge_frames_to_video
from head3D_fuse.visualization.vis_utils import (
    _save_frame_fuse_3dkpt_visualization,
    _save_view_visualizations,
    visualizer,
)

# fuse
from head3D_fuse.fuse.fuse import fuse_3view_keypoints

# temporal smooth
from head3D_fuse.smooth.temporal_smooth import (
    smooth_keypoints_sequence,
)

# comparison
from head3D_fuse.smooth.compare_fused_smoothed import KeypointsComparator
from head3D_fuse.fuse.compare_fused import FusedViewComparator

logger = logging.getLogger(__name__)
VALID_ALIGNMENT_METHODS = ("none", "procrustes", "procrustes_trimmed")

# 定义需要保留的关键点索引：头部 + 肩部/颈部 + 双手
KEEP_KEYPOINT_INDICES = (
    # 头部: 鼻子、眼睛、耳朵
    list(range(0, 5))  # 0-4: nose, left-eye, right-eye, left-ear, right-ear
    # 肩部和颈部
    + [5, 6]  # left-shoulder, right-shoulder
    # 双手（包括手腕）
    + list(range(21, 63))  # 21-62: 右手(21-41) + 左手(42-62)
    # 肩峰和颈部
    + [67, 68, 69]  # left-acromion, right-acromion, neck
)


# =====================================================================
# Fuse 处理函数
# =====================================================================
def _fuse_single_person_env(
    person_env_dir: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
    frame_triplets: list,
    view_list: list,
) -> list:
    """融合多视图关键点

    Args:
        person_env_dir: 人员环境目录
        out_root: 输出根目录
        infer_root: 推理根目录
        cfg: 配置
        frame_triplets: 帧三元组列表
        view_list: 视图列表

    Returns:
        fused_frame_indices: 成功融合并保存的帧索引列表
    """
    person_id = person_env_dir.parent.name
    env_name = person_env_dir.name

    fused_frame_indices = []
    diff_reports = []
    fused_method = cfg.fuse.get("fuse_method", "median")

    logger.info(f"==== Starting Fuse for Person: {person_id}, Env: {env_name} ====")

    # 随机保存100frame作为可视化内容
    random_save_idx = np.random.uniform(0, len(frame_triplets), 100).astype(int)

    for i, triplet in enumerate(
        tqdm(frame_triplets, desc=f"Fusing {person_id}/{env_name}")
    ):
        diff = compare_npz_files(triplet.npz_paths)
        if diff:
            diff_reports.append(diff)

        outputs = {
            view: load_npz_output(npz_path)
            for view, npz_path in triplet.npz_paths.items()
        }
        keypoints_by_view = {}
        for view in view_list:
            filtered_view_3dkpt = _normalize_keypoints(
                outputs[view].get("pred_keypoints_3d")
            )
            filtered_view_2dkpt = _normalize_keypoints(
                outputs[view].get("pred_keypoints_2d")
            )
            keypoints_by_view[view] = outputs[view]["pred_keypoints_3d"]
            outputs[view]["filtered_pred_keypoints_3d"] = filtered_view_3dkpt
            outputs[view]["filtered_pred_keypoints_2d"] = filtered_view_2dkpt

        # 检查是否有包含NaN的视角
        missing_views = [
            view for view, kpt in keypoints_by_view.items() if np.all(np.isnan(kpt))
        ]
        if missing_views:
            logger.warning(
                "Missing pred_keypoints_3d for frame %s in %s/%s (views: %s)",
                triplet.frame_idx,
                person_id,
                env_name,
                ",".join(missing_views),
            )
            continue

        view_transforms = cfg.fuse.get("view_transforms")
        transform_mode = cfg.fuse.get("transform_mode", "world_to_camera")
        alignment_method = cfg.fuse.get("alignment_method", "none")
        alignment_reference = cfg.fuse.get("alignment_reference")
        alignment_scale = cfg.fuse.get("alignment_scale", True)
        alignment_trim_ratio = cfg.fuse.get("alignment_trim_ratio", 0.2)
        alignment_max_iters = cfg.fuse.get("alignment_max_iters", 3)

        # 融合三个视角
        fused_kpt, fused_mask, n_valid = fuse_3view_keypoints(
            keypoints_by_view,
            method=fused_method,
            view_transforms=view_transforms,
            transform_mode=transform_mode,
            alignment_method=alignment_method,
            alignment_reference=alignment_reference,
            alignment_scale=alignment_scale,
            alignment_trim_ratio=alignment_trim_ratio,
            alignment_max_iters=alignment_max_iters,
        )

        fused_frame_indices.append(triplet.frame_idx)
        # 保存融合的关键点
        save_dir = infer_root / person_id / env_name / "fused_npz"
        _save_fused_keypoints(
            save_dir=save_dir,
            frame_idx=triplet.frame_idx,
            fused_keypoints=fused_kpt,
            fused_mask=fused_mask,
            n_valid=n_valid,
            npz_paths=triplet.npz_paths,
        )

        # 可视化结果
        if i in random_save_idx:
            # 单独保存三个视角的可视化结果
            if cfg.visualize.save_single_view_visualization:
                for view in view_list:
                    if view not in outputs:
                        logger.warning(
                            f"Missing output for view={view} frame={triplet.frame_idx}"
                        )
                        continue
                    vis_root = (
                        out_root / person_id / env_name / "fused" / "different_vis"
                    )
                    _save_view_visualizations(
                        output=outputs[view],
                        save_root=vis_root,
                        view=view,
                        frame_idx=triplet.frame_idx,
                        cfg=cfg,
                        visualizer=visualizer,
                    )

            # 保存三个视角的frame和融合结果的可视化
            if cfg.visualize.save_fused_view_visualization:
                _save_frame_fuse_3dkpt_visualization(
                    save_dir=out_root / person_id / env_name / "fused" / "vis_together",
                    frame_idx=triplet.frame_idx,
                    fused_keypoints=fused_kpt,
                    outputs=outputs,
                    visualizer=visualizer,
                )

    # 保存 npz diff 报告
    if diff_reports:
        diff_path = out_root / person_id / env_name / "npz_diff_report.json"
        diff_path.parent.mkdir(parents=True, exist_ok=True)
        with diff_path.open("w", encoding="utf-8") as f:
            json.dump(diff_reports, f, ensure_ascii=False, indent=2)
        logger.info("Saved npz diff report to %s", diff_path)
    else:
        logger.info("No npz differences found for %s/%s", person_id, env_name)

    # 融合frame到video
    # ! 因为现在只保存100frame，所以也没办法把所有的frame保存成video了
    # merge_frames_to_video(
    #     frame_dir=out_root / person_id / env_name / "fused" / "vis_together",
    #     output_video_path=out_root
    #     / person_id
    #     / env_name
    #     / "merged_video"
    #     / "fused_3d_keypoints.mp4",
    #     fps=30,
    # )
    # merge_frames_to_video(
    #     frame_dir=out_root / person_id / env_name / "fused" / "different_vis" / "front",
    #     output_video_path=out_root
    #     / person_id
    #     / env_name
    #     / "merged_video"
    #     / "front.mp4",
    #     fps=30,
    # )
    # merge_frames_to_video(
    #     frame_dir=out_root / person_id / env_name / "fused" / "different_vis" / "left",
    #     output_video_path=out_root / person_id / env_name / "merged_video" / "left.mp4",
    #     fps=30,
    # )
    # merge_frames_to_video(
    #     frame_dir=out_root / person_id / env_name / "fused" / "different_vis" / "right",
    #     output_video_path=out_root
    #     / person_id
    #     / env_name
    #     / "merged_video"
    #     / "right.mp4",
    #     fps=30,
    # )

    logger.info(f"==== Finished Fuse for Person: {person_id}, Env: {env_name} ====")

    return fused_frame_indices


def _load_fused_keypoints_from_file(
    infer_root: Path, person_id: str, env_name: str, frame_idx: int
) -> Optional[np.ndarray]:
    """从 fused_npz 目录读取指定帧的融合关键点。"""
    fused_path = infer_root / person_id / env_name / "fused_npz" / f"frame_{frame_idx:06d}_fused.npy"
    if not fused_path.exists():
        logger.warning("Missing fused file for frame %s: %s", frame_idx, fused_path)
        return None

    payload = np.load(fused_path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        payload = payload.item()

    if not isinstance(payload, dict):
        logger.warning("Invalid fused payload format: %s", fused_path)
        return None

    fused_keypoints = payload.get("fused_keypoints_3d")
    if not isinstance(fused_keypoints, np.ndarray):
        logger.warning("Missing fused_keypoints_3d in fused file: %s", fused_path)
        return None

    return cast(np.ndarray, fused_keypoints)


# =====================================================================
# Smooth 处理函数
# =====================================================================
def _smooth_fused_keypoints_env(
    person_env_dir: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
    frame_triplets: list,
    fused_frame_indices: list,
    view_list: list,
) -> tuple:
    """平滑融合后的关键点

    Args:
        person_env_dir: 人员环境目录
        out_root: 输出根目录
        infer_root: 推理根目录
        cfg: 配置
        frame_triplets: 帧三元组列表
        fused_frame_indices: 成功融合并保存的帧索引列表
        view_list: 视图列表

    Returns:
        keypoints_array: 原始关键点数组
        smoothed_array: 平滑后的关键点数组
        sorted_frames: 成功读取并参与处理的帧索引
    """
    person_id = person_env_dir.parent.name
    env_name = person_env_dir.name

    if not fused_frame_indices:
        logger.info("No fused keypoints to smooth")
        return None, None, []

    # 1. 从 fused_npz 读取并组装 numpy 数组 (T, N, 3)
    sorted_frames = sorted(set(int(idx) for idx in fused_frame_indices))
    loaded_frames = []
    loaded_keypoints = []
    for frame_idx in sorted_frames:
        fused_kpt = _load_fused_keypoints_from_file(
            infer_root=infer_root,
            person_id=person_id,
            env_name=env_name,
            frame_idx=frame_idx,
        )
        if fused_kpt is None:
            continue
        loaded_frames.append(frame_idx)
        loaded_keypoints.append(fused_kpt)

    if not loaded_keypoints:
        logger.warning("No fused keypoints loaded from disk for %s/%s", person_id, env_name)
        return None, None, []

    sorted_frames = loaded_frames
    keypoints_array = np.stack(loaded_keypoints, axis=0)
    logger.info(f"Keypoints array shape: {keypoints_array.shape}")

    # 检查是否启用平滑
    if not cfg.smooth.get("enable_temporal_smooth", False):
        logger.info("Temporal smoothing is disabled")
        return keypoints_array, None, sorted_frames

    logger.info(f"Applying temporal smoothing to {len(sorted_frames)} frames...")

    # 2. 根据方法准备参数
    smooth_method = cfg.smooth.get("temporal_smooth_method", "gaussian")
    smooth_kwargs = {}

    if smooth_method == "gaussian":
        smooth_kwargs["sigma"] = cfg.smooth.get("temporal_smooth_sigma", 1.5)
    elif smooth_method == "savgol":
        smooth_kwargs["window_length"] = cfg.smooth.get(
            "temporal_smooth_window_length", 11
        )
        smooth_kwargs["polyorder"] = cfg.smooth.get("temporal_smooth_polyorder", 3)
    elif smooth_method == "kalman":
        smooth_kwargs["process_variance"] = cfg.smooth.get(
            "temporal_smooth_process_variance", 1e-5
        )
        smooth_kwargs["measurement_variance"] = cfg.smooth.get(
            "temporal_smooth_measurement_variance", 1e-2
        )
    elif smooth_method == "bilateral":
        smooth_kwargs["sigma_space"] = cfg.smooth.get(
            "temporal_smooth_sigma_space", 1.5
        )
        smooth_kwargs["sigma_range"] = cfg.smooth.get(
            "temporal_smooth_sigma_range", 0.1
        )

    # 3. 执行平滑
    smoothed_array = smooth_keypoints_sequence(
        keypoints=keypoints_array, method=smooth_method, **smooth_kwargs
    )
    logger.info(f"Smoothed keypoints shape: {smoothed_array.shape}")

    # 随机保存100frame作为可视化内容
    random_save_idx = np.random.uniform(0, len(sorted_frames), 100).astype(int)

    frame_to_triplet = {triplet.frame_idx: triplet for triplet in frame_triplets}

    # 4. 保存平滑后的结果
    for i, frame_idx in enumerate(sorted_frames):
        smooth_fused_kpt = smoothed_array[i]

        save_dir = infer_root / person_id / env_name / "smoothed_fused_npz"
        _save_fused_keypoints(
            save_dir=save_dir,
            frame_idx=frame_idx,
            fused_keypoints=smooth_fused_kpt,
            fused_mask=np.ones((smooth_fused_kpt.shape[0],), dtype=np.bool_),
            n_valid=smooth_fused_kpt.shape[0],
            npz_paths={},
        )

        # 使用该帧对应的outputs进行可视化
        # * 这里也只选100frame进行可视化
        if i in random_save_idx:
            triplet = frame_to_triplet.get(frame_idx)
            if triplet is not None:
                frame_outputs = {
                    view: load_npz_output(npz_path)
                    for view, npz_path in triplet.npz_paths.items()
                    if view in view_list
                }
                _save_frame_fuse_3dkpt_visualization(
                    save_dir=out_root
                    / person_id
                    / env_name
                    / "smoothed"
                    / "smoothed_fused"
                    / "vis_together",
                    frame_idx=frame_idx,
                    fused_keypoints=smooth_fused_kpt,
                    outputs=frame_outputs,
                    visualizer=visualizer,
                )
            else:
                logger.warning(
                    f"No triplet found for frame {frame_idx} during smoothing visualization"
                )

    logger.info(f"✓ Temporal smoothing completed and saved {len(sorted_frames)} frames")

    # merge frame to video
    # ! 因为现在只可视化100frame，所以也没办法合成video了
    # merge_frames_to_video(
    #     frame_dir=out_root
    #     / person_id
    #     / env_name
    #     / "smoothed"
    #     / "smoothed_fused"
    #     / "vis_together",
    #     output_video_path=out_root
    #     / person_id
    #     / env_name
    #     / "merged_video"
    #     / "smoothed_fused_3d_keypoints.mp4",
    #     fps=30,
    # )

    logger.info(
        f"==== Finished Temporal Smoothing for Person: {person_id}, Env: {env_name} ===="
    )

    return keypoints_array, smoothed_array, sorted_frames


def _normalize_keypoints(keypoints: Optional[np.ndarray]) -> np.ndarray:
    """归一化关键点并过滤只保留头部、肩部和双手的关键点。

    Args:
        keypoints: 输入的关键点数组，形状可能是 (batch, N, 3) 或 (N, 3)

    Returns:
        过滤后的关键点数组，形状为 (M, 3)，其中M是保留的关键点数量
        如果输入为None，返回填充NaN的数组
    """
    num_keep_points = len(KEEP_KEYPOINT_INDICES)

    if keypoints is None:
        # 当关键点缺失时，创建填充NaN的数组
        return np.full((num_keep_points, 3), np.nan, dtype=np.float32)

    # 明确类型以避免类型检查错误
    kpt_array = cast(np.ndarray, np.asarray(keypoints))
    assert kpt_array is not None  # 帮助类型检查器

    # 处理batch维度
    if kpt_array.ndim == 3 and kpt_array.shape[0] >= 1:
        kpt_array = kpt_array[0]

    # 过滤关键点，只保留头部、肩部和双手
    if kpt_array.shape[0] > max(KEEP_KEYPOINT_INDICES):
        filtered_keypoints = kpt_array[KEEP_KEYPOINT_INDICES]
    else:
        # 如果关键点数量不足，填充NaN
        logger.warning(
            "Keypoints shape %s is smaller than expected, padding with NaN",
            kpt_array.shape,
        )
        filtered_keypoints = np.full((num_keep_points, 3), np.nan, dtype=np.float32)
        # 复制可用的关键点
        available_indices = [i for i in KEEP_KEYPOINT_INDICES if i < kpt_array.shape[0]]
        for new_idx, old_idx in enumerate(available_indices):
            if new_idx < num_keep_points:
                filtered_keypoints[new_idx] = kpt_array[old_idx]

    return filtered_keypoints


def _filter_keypoints_sequence(keypoints_seq: np.ndarray) -> np.ndarray:
    """过滤时序关键点，输出统一为 (T, M, 3)。"""
    seq = np.asarray(keypoints_seq)
    if seq.ndim != 3 or seq.shape[-1] < 3:
        raise ValueError(
            f"Expected keypoints sequence shape (T, N, 3), got {seq.shape}"
        )
    return np.stack([_normalize_keypoints(frame) for frame in seq], axis=0)


def _resolve_filtered_keypoint_indices(
    requested_indices: Optional[list],
    filtered_num_points: int,
    context_name: str,
) -> list:
    """将配置中的关键点索引解析为过滤后数组中的索引。"""
    if filtered_num_points <= 0:
        return []

    if requested_indices is None:
        return list(range(min(7, filtered_num_points)))

    keep_to_filtered = {
        original_idx: filtered_idx
        for filtered_idx, original_idx in enumerate(KEEP_KEYPOINT_INDICES)
    }
    resolved = []
    seen = set()

    for idx in requested_indices:
        if not isinstance(idx, (int, np.integer)):
            logger.warning(
                "Ignored non-integer keypoint index in %s: %s", context_name, idx
            )
            continue

        idx_int = int(idx)

        # 兼容已是过滤后索引的配置
        if 0 <= idx_int < filtered_num_points:
            mapped_idx = idx_int
        # 兼容仍使用原始关键点索引的配置（映射到过滤后位置）
        elif idx_int in keep_to_filtered:
            mapped_idx = keep_to_filtered[idx_int]
            if mapped_idx >= filtered_num_points:
                logger.warning(
                    "Ignored keypoint index %s in %s after mapping (mapped=%s, max=%s)",
                    idx_int,
                    context_name,
                    mapped_idx,
                    filtered_num_points - 1,
                )
                continue
        else:
            logger.warning(
                "Ignored keypoint index %s in %s (not present in filtered keypoints)",
                idx_int,
                context_name,
            )
            continue

        if mapped_idx not in seen:
            seen.add(mapped_idx)
            resolved.append(mapped_idx)

    if not resolved:
        logger.warning(
            "No valid keypoint indices resolved in %s; fallback to first %d filtered keypoints",
            context_name,
            min(7, filtered_num_points),
        )
        return list(range(min(7, filtered_num_points)))

    return resolved


# =====================================================================
# Comparison 处理函数
# =====================================================================
def _compare_fused_smoothed_keypoints(
    person_env_dir: Path,
    out_root: Path,
    cfg: DictConfig,
    keypoints_array: np.ndarray,
    smoothed_array: np.ndarray,
):
    """比较平滑前后的关键点差异

    Args:
        person_env_dir: 人员环境目录
        out_root: 输出根目录
        cfg: 配置
        keypoints_array: 原始关键点数组
        smoothed_array: 平滑后的关键点数组
    """
    if keypoints_array is None or smoothed_array is None:
        logger.info("Comparison is disabled or no data to compare")
        return

    person_id = person_env_dir.parent.name
    env_name = person_env_dir.name

    if not cfg.smooth.get("enable_comparison", False):
        logger.info("Comparison is disabled or no data to compare")
        return

    logger.info("=" * 70)
    logger.info("Comparing fused and smoothed keypoints...")
    logger.info("=" * 70)

    try:
        # 统一比较为过滤后的关键点
        filtered_keypoints_array = _filter_keypoints_sequence(keypoints_array)
        filtered_smoothed_array = _filter_keypoints_sequence(smoothed_array)

        # 创建比较器
        comparator = KeypointsComparator(
            filtered_keypoints_array, filtered_smoothed_array
        )

        # 获取要评估的关键点索引
        requested_indices = cfg.smooth.get("comparison_keypoint_indices")
        keypoint_indices = _resolve_filtered_keypoint_indices(
            requested_indices=requested_indices,
            filtered_num_points=filtered_keypoints_array.shape[1],
            context_name="smooth.comparison_keypoint_indices",
        )

        # 计算所有指标（按索引过滤）
        metrics = comparator.compute_metrics(keypoint_indices=keypoint_indices)
        logger.info(f"Computed {len(metrics)} metrics for keypoints {keypoint_indices}")

        # 设置输出目录
        comparison_dir = out_root / person_id / env_name / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存指标到 JSON
        metrics_path = comparison_dir / "smoothing_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info(f"✓ Saved metrics to {metrics_path}")

        # 2. 生成并保存详细报告（按索引）
        report_path = comparison_dir / "smoothing_comparison_report.txt"
        report = comparator.generate_report(
            save_path=report_path, keypoint_indices=keypoint_indices
        )
        logger.info(f"✓ Saved report to {report_path}")

        # 打印关键指标到日志
        logger.info("")
        logger.info("Key Metrics Summary:")
        logger.info(f"  Mean Difference:       {metrics['mean_difference']:.6f}")
        logger.info(f"  Jitter Reduction:      {metrics['jitter_reduction']:.2f}%")
        logger.info(
            f"  Acceleration Reduction: {metrics['acceleration_reduction']:.2f}%"
        )
        logger.info("")

        # 3. 生成可视化图表（如果配置启用）
        if cfg.smooth.get("enable_comparison_plots", True):
            logger.info("Generating comparison plots...")

            # 轨迹对比图（显示0-6关键点的X、Y、Z）
            trajectory_plot_path = comparison_dir / "trajectory_comparison.png"
            comparator.plot_comparison(
                save_path=trajectory_plot_path, keypoint_indices=keypoint_indices
            )
            logger.info(f"✓ Saved trajectory plot to {trajectory_plot_path}")
            logger.info(f"  Visualized keypoints: {keypoint_indices}")

            # 指标对比图（按索引过滤）
            metrics_plot_path = comparison_dir / "metrics_comparison.png"
            comparator.plot_metrics(
                save_path=metrics_plot_path, keypoint_indices=keypoint_indices
            )
            logger.info(f"✓ Saved metrics plot to {metrics_plot_path}")

        logger.info("=" * 70)
        logger.info("✓ Comparison completed successfully")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to generate comparison report: {e}", exc_info=True)


# =====================================================================
# Fused vs Views Comparison 处理函数
# =====================================================================
def _compare_fused_with_views(
    person_env_dir: Path,
    out_root: Path,
    cfg: DictConfig,
    keypoints_array: Optional[np.ndarray],
    frame_triplets: list,
    fused_frame_indices: list,
    view_list: list,
):
    """比较融合结果与各个单视角的3D关键点

    Args:
        person_env_dir: 人员环境目录
        out_root: 输出根目录
        cfg: 配置
        keypoints_array: 融合后的关键点数组 (T, N, 3)
        frame_triplets: 帧三元组列表
        fused_frame_indices: 与 keypoints_array 对齐的帧索引列表
        view_list: 视角列表
    """
    person_id = person_env_dir.parent.name
    env_name = person_env_dir.name

    # 检查是否启用对比
    if (
        keypoints_array is None
        or not fused_frame_indices
        or not cfg.fuse.get("enable_fused_view_comparison", False)
    ):
        logger.info("Fused vs Views comparison is disabled or no data to compare")
        return

    logger.info("=" * 70)
    logger.info(
        f"Comparing fused keypoints with single-view keypoints for {person_id}/{env_name}"
    )
    logger.info("=" * 70)

    try:
        # 1. 准备数据：将字典转换为numpy数组
        sorted_frames = [int(idx) for idx in fused_frame_indices]

        # 融合后的关键点 (T, M, 3) - 使用过滤后关键点
        fused_array = _filter_keypoints_sequence(keypoints_array)
        logger.info(f"Fused keypoints shape: {fused_array.shape}")

        frame_to_triplet = {triplet.frame_idx: triplet for triplet in frame_triplets}

        # 各视角的关键点
        view_keypoints = {}
        for view in view_list:
            view_kpts_list = []
            for frame_idx in sorted_frames:
                triplet = frame_to_triplet.get(frame_idx)
                if triplet is None or view not in triplet.npz_paths:
                    logger.warning(
                        f"Missing triplet data for frame {frame_idx}, view {view}"
                    )
                    continue

                frame_output = load_npz_output(triplet.npz_paths[view])
                kpts_3d = _normalize_keypoints(frame_output.get("pred_keypoints_3d"))

                view_kpts_list.append(kpts_3d)

            if len(view_kpts_list) == len(sorted_frames):
                view_keypoints[view] = np.stack(view_kpts_list, axis=0)
                logger.info(
                    f"View '{view}' keypoints shape: {view_keypoints[view].shape}"
                )
            else:
                logger.warning(
                    f"Incomplete data for view '{view}': {len(view_kpts_list)}/{len(sorted_frames)} frames"
                )

        if len(view_keypoints) < 2:
            logger.error("Not enough view data for comparison (need at least 2 views)")
            return

        # 2. 创建比较器
        comparator = FusedViewComparator(fused_array, view_keypoints)

        # 3. 获取要评估的关键点索引
        requested_indices = cfg.fuse.get("fused_view_comparison_keypoint_indices")
        keypoint_indices = _resolve_filtered_keypoint_indices(
            requested_indices=requested_indices,
            filtered_num_points=fused_array.shape[1],
            context_name="fuse.fused_view_comparison_keypoint_indices",
        )

        # 4. 计算指标
        metrics = comparator.compute_metrics(keypoint_indices=keypoint_indices)
        logger.info(f"Computed metrics for {len(keypoint_indices)} keypoints")

        # 5. 设置输出目录
        comparison_dir = out_root / person_id / env_name / "fused_vs_views_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # 6. 保存JSON指标
        metrics_path = comparison_dir / "fused_vs_views_metrics.json"
        comparator.export_metrics_json(metrics_path, keypoint_indices=keypoint_indices)
        logger.info(f"✓ Saved metrics to {metrics_path}")

        # 7. 生成并保存文本报告
        report_path = comparison_dir / "fused_vs_views_report.txt"
        report = comparator.generate_report(
            save_path=report_path, keypoint_indices=keypoint_indices
        )
        logger.info(f"✓ Saved report to {report_path}")

        # 8. 打印关键指标摘要
        logger.info("")
        logger.info("Key Metrics Summary:")
        logger.info("  Mean distance to views:")
        for view, dist in metrics["mean_distance_to_views"].items():
            logger.info(f"    {view}: {dist:.6f}")
        logger.info(
            f"  Distance to centroid: {metrics['mean_distance_to_centroid']:.6f}"
        )
        logger.info(f"  Fused jitter: {metrics['fused_jitter']['mean']:.6f}")
        logger.info("")

        # 9. 生成可视化图表（如果配置启用）
        if cfg.fuse.get("enable_fused_view_comparison_plots", True):
            logger.info("Generating fused vs views comparison plots...")
            plot_path = comparison_dir / "fused_vs_views_comparison.png"
            comparator.plot_comparison(
                save_path=plot_path, keypoint_indices=keypoint_indices
            )
            logger.info(f"✓ Saved comparison plot to {plot_path}")

        logger.info("=" * 70)
        logger.info("✓ Fused vs Views comparison completed successfully")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Failed to compare fused with views: {e}", exc_info=True)


# =====================================================================
# 核心处理逻辑：处理单个人的数据
# =====================================================================
def process_single_person_env(
    person_env_dir: Path,
    out_root: Path,
    infer_root: Path,
    cfg: DictConfig,
):
    """处理单个人员的所有环境和视角"""

    person_id = person_env_dir.parent.name
    env_name = person_env_dir.name
    view_list = cfg.infer.get("view_list")

    if view_list is None:
        view_list = ["front", "left", "right"]

    annotation_dict = get_annotation_dict(cfg.paths.start_mid_end_path)

    logger.info(f"==== Starting Process for Person: {person_id}, Env: {env_name} ====")

    frame_triplets, report = assemble_view_npz_paths(
        person_env_dir, view_list, annotation_dict
    )
    if not frame_triplets:
        logger.warning(f"No aligned frames found for {person_id}/{env_name}")
        return

    # 1. 融合多视图关键点
    fused_frame_indices = _fuse_single_person_env(
        person_env_dir=person_env_dir,
        out_root=out_root,
        infer_root=infer_root,
        cfg=cfg,
        frame_triplets=frame_triplets,
        view_list=view_list,
    )

    # 2. 平滑融合后的关键点
    keypoints_array, smoothed_array, fused_frame_indices = _smooth_fused_keypoints_env(
        person_env_dir=person_env_dir,
        out_root=out_root,
        infer_root=infer_root,
        cfg=cfg,
        frame_triplets=frame_triplets,
        fused_frame_indices=fused_frame_indices,
        view_list=view_list,
    )

    # 3. 比较平滑前后的差异
    _compare_fused_smoothed_keypoints(
        person_env_dir=person_env_dir,
        out_root=out_root,
        cfg=cfg,
        keypoints_array=keypoints_array,
        smoothed_array=smoothed_array,
    )

    # 4. 比较融合结果与各单视角
    _compare_fused_with_views(
        person_env_dir=person_env_dir,
        out_root=out_root,
        cfg=cfg,
        keypoints_array=keypoints_array,
        frame_triplets=frame_triplets,
        fused_frame_indices=fused_frame_indices,
        view_list=view_list,
    )

    # * 结束推理一个人的一个环境之后，清空内存
    # 释放当前 person/env 的大对象，避免长流程中内存持续增长
    del fused_frame_indices
    del keypoints_array
    del smoothed_array
    del frame_triplets
    del report

    # 触发 Python 垃圾回收
    gc.collect()
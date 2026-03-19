#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Triangulate 3D joints from two-view 2D keypoints using either video frames or pre-extracted keypoints.
Supports modular triangulation, pose estimation, and interactive 3D visualization.

Author: Kaixu Chen
Last Modified: August 4th, 2025
"""

import os
import numpy as np
import cv2
import logging
import hydra
import traceback

import multiprocessing as mp
from pathlib import Path

logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

from mesh_triangulation.camera_position_mapping import prepare_camera_position
from mesh_triangulation.load import load_mesh_from_npz
from mesh_triangulation.multi_triangulation import (
    triangulate_with_missing,
    smooth_combo,
)
from mesh_triangulation.mesh_3d_eval import (
    evaluate_face3d_pro,
    export_report,
    visualize_metrics,
)
from mesh_triangulation.interpolate_missing import (
    interpolate_missing_frames,
)

# vis
from mesh_triangulation.vis.frame_visualization import draw_and_save_mesh_from_frame
from mesh_triangulation.vis.mesh_visualization import visualize_3d_mesh
from mesh_triangulation.vis.merge_video import merge_frames_to_video

# save
from mesh_triangulation.save import save_3d_joints

from logging.handlers import RotatingFileHandler


def _setup_worker_logger(log_root: Path, level=logging.INFO):
    log_root.mkdir(parents=True, exist_ok=True)
    proc = mp.current_process()
    # TODO：这里最好能改成环境-人物的文件名字
    fname = f"worker-{proc.name}-pid{os.getpid()}.log"  # 例如 worker-ForkPoolWorker-1-pid12345.log
    fpath = log_root / fname

    logger = logging.getLogger()  # root 或者你项目的主 logger 名称
    logger.setLevel(level)

    # 清掉已有文件 handler（避免重复添加）
    for h in list(logger.handlers):
        if isinstance(h, (logging.FileHandler, RotatingFileHandler)):
            logger.removeHandler(h)
            h.close()

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(processName)s pid=%(process)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(fpath, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)
    logger.addHandler(fh)

    # 也可保留控制台输出（可选）
    # sh = logging.StreamHandler()
    # sh.setFormatter(fmt); sh.setLevel(level)
    # logger.addHandler(sh)

    logger.info(f"[LOGGER] process logger set to {fpath}")


def resize_frame_and_mesh(frame: np.ndarray, mesh: np.ndarray, new_size):
    """
    同时缩放图像 frame 和 mesh 坐标
    参数:
      frame: np.ndarray, 形状 (H, W, 3)
      mesh: np.ndarray, 形状 (N, 3)
      new_size: (new_w, new_h)
    返回:
      frame_resized, mesh_rescaled
    """
    new_w, new_h = new_size

    # 图像缩放
    if frame.shape[1] != new_w and frame.shape[0] != new_h:

        frame_resized = cv2.resize(
            frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )
    else:
        frame_resized = frame.copy()

    # normalize mesh 坐标
    if not np.allclose(mesh, 0):
        mesh_normalized = np.asarray(mesh, dtype=np.float32).copy()
        mesh_normalized[:, 0] *= new_w
        mesh_normalized[:, 1] *= new_h  # y 方向缩放
        # z 一般保持不变（除非是相机空间深度）
    else:
        mesh_normalized = mesh.copy()

    return frame_resized, mesh_normalized


def detail_to_txt(detail: dict, save_path: Path):
    """
    把 triangulate_with_missing 的 detail 结果整理成可写入 txt 的格式
    detail: triangulate_with_missing 返回的第三个字典
    save_path: 输出 txt 路径
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if detail is None:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("No triangulation detail available.\n")
        return

    lines = []
    views = detail["views_order"]
    used_mask = detail["used_mask"]
    used_weights = detail["used_weights"]
    used_pts2d = detail["used_pts2d"]
    per_point = detail["per_point"]

    lines.append(f"# Triangulation detail report")
    lines.append(f"# Total points: {len(per_point)}")
    lines.append(f"# Views: {', '.join(views)}")
    lines.append("-" * 80)

    for j, info in enumerate(per_point):
        if info is None:
            lines.append(f"[{j:04d}] Status: <None>\n")
            continue

        lines.append(f"[{j:04d}] Status: {info.get('status', 'unknown')}")
        lines.append(f"  Views used: {', '.join(info.get('views', []))}")
        lines.append(f"  Num views: {info.get('num_views', 0)}")

        if "reproj_err" in info:
            lines.append(
                f"  Reproj error: {info['reproj_err']:.3f}px (thr={info.get('thr', 'NA')})"
            )
        if "max_sampson" in info:
            lines.append(
                f"  Max Sampson: {info['max_sampson']:.3f} (thr={info.get('thr', 'NA')})"
            )

        # --- 每个视角的坐标与权重 ---
        lines.append(f"  2D points:")
        for v_idx, v in enumerate(views):
            if used_mask[j, v_idx]:
                x, y = used_pts2d[j, v_idx]
                w = used_weights[j, v_idx]
                lines.append(f"    - {v}: ({x:.2f}, {y:.2f}), weight={w:.2f}")
        lines.append("-" * 80)

    txt = "\n".join(lines)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[INFO] Saved triangulation detail to: {save_path}")


# ---------- 主处理函数 ----------
def process_one_frame(
    f_frame,
    f_mesh,
    l_frame,
    l_mesh,
    r_frame,
    r_mesh,
    rt_info,
    K,
    output_path: Path,
    i: int,
    vis,
):
    # * 确保三视点图像尺寸一致
    if (
        f_frame.shape[0:2] != l_frame.shape[0:2]
        or f_frame.shape[0:2] != r_frame.shape[0:2]
        or l_frame.shape[0:2] != r_frame.shape[0:2]
    ):

        # 统一目标尺寸（取最大值）
        target_h = max(f_frame.shape[0], l_frame.shape[0], r_frame.shape[0])
        target_w = max(f_frame.shape[1], l_frame.shape[1], r_frame.shape[1])

        f_frame_resized, f_mesh_unnormalized = resize_frame_and_mesh(
            f_frame, f_mesh, (target_w, target_h)
        )
        l_frame_resized, l_mesh_unnormalized = resize_frame_and_mesh(
            l_frame, l_mesh, (target_w, target_h)
        )
        r_frame_resized, r_mesh_unnormalized = resize_frame_and_mesh(
            r_frame, r_mesh, (target_w, target_h)
        )
    else:
        target_h, target_w = f_frame.shape[0:2]

        f_frame_resized, f_mesh_unnormalized = f_frame, f_mesh
        l_frame_resized, l_mesh_unnormalized = l_frame, l_mesh
        r_frame_resized, r_mesh_unnormalized = r_frame, r_mesh

    observations = {
        "front": f_mesh_unnormalized[:, :2],
        "left": l_mesh_unnormalized[:, :2],
        "right": r_mesh_unnormalized[:, :2],
    }

    mesh_3d, stats, detail = triangulate_with_missing(
        observations=observations,
        extrinsics=rt_info,
        Ks=K,
        max_err_px=800.0,
        sampson_px=200.0,
    )

    if np.isnan(mesh_3d).all():
        logger.warning(f"Triangulation failed for frame {i}")

        # 保存3D关键点构成信息
        detail_path = output_path / "frame_triangulation_detail" / f"frame_{i:04d}.txt"
        detail_to_txt(detail, detail_path)

        # 保存统计信息
        stats_path = output_path / "frame_triangulation_stats.txt"
        with open(stats_path, "a") as f:
            f.write(f"Frame {i}: Triangulation failed.\n")

        return False, (target_h, target_w)

    else:
        logger.info(
            f"Frame {i}: Triangulation stats: {stats['ok']}/{stats['total']} points OK, "
            f"Mean RPE: {stats['mean_rpe']:.2f}px, "
            f"Epi Inlier: {stats['epi_inlier']*100:.2f}%, "
            f"Cheirality: {stats['cheirality']*100:.2f}%"
        )

        # 保存3D关键点构成信息
        detail_path = output_path / "frame_triangulation_detail" / f"frame_{i:04d}.txt"
        detail_to_txt(detail, detail_path)

        # 保存统计信息
        stats_path = output_path / "frame_triangulation_stats.txt"
        with open(stats_path, "a") as f:
            f.write(
                f"Frame {i}: Triangulation stats: {stats['ok']}/{stats['total']} points OK, "
                f"Mean RPE: {stats['mean_rpe']:.2f}px, "
                f"Epi Inlier: {stats['epi_inlier']*100:.2f}%, "
                f"Cheirality: {stats['cheirality']*100:.2f}%\n"
            )

        if vis.save_mesh_frame:
            draw_and_save_mesh_from_frame(
                frame=f_frame_resized,
                mesh=f_mesh_unnormalized,
                save_path=output_path / "vis" / f"mesh_frames/front/frame_{i:04d}.png",
                color=(0, 255, 0),
                radius=2,
                draw_tesselation=True,
                draw_contours=True,
                with_index=False,
            )
            draw_and_save_mesh_from_frame(
                frame=l_frame_resized,
                mesh=l_mesh_unnormalized,
                save_path=output_path / "vis" / f"mesh_frames/left/frame_{i:04d}.png",
                color=(0, 255, 0),
                radius=2,
                draw_tesselation=True,
                draw_contours=True,
                with_index=False,
            )
            draw_and_save_mesh_from_frame(
                frame=r_frame_resized,
                mesh=r_mesh_unnormalized,
                save_path=output_path / "vis" / f"mesh_frames/right/frame_{i:04d}.png",
                color=(0, 255, 0),
                radius=2,
                draw_tesselation=True,
                draw_contours=True,
                with_index=False,
            )

        return mesh_3d, (target_h, target_w)


def process_one_video(
    environment_dir: dict[str, Path], output_path: Path, rt_info, K, vis
):

    output_path.mkdir(parents=True, exist_ok=True)

    front_frames, front_mesh, front_video_info, front_not_mesh_list = (
        load_mesh_from_npz(environment_dir["front"])
    )
    if front_frames is None:
        logger.warning(f"No front view data found in {environment_dir['front']}")
        return

    left_frames, left_mesh, left_video_info, left_not_mesh_list = load_mesh_from_npz(
        environment_dir["left"]
    )
    right_frames, right_mesh, right_video_info, right_not_mesh_list = (
        load_mesh_from_npz(environment_dir["right"])
    )

    # * 确保三视点帧数一致
    if (
        left_frames.shape[0] != right_frames.shape[0]
        or left_frames.shape[0] != front_frames.shape[0]
        or right_frames.shape[0] != front_frames.shape[0]
    ):
        min_frames = min(
            left_frames.shape[0], right_frames.shape[0], front_frames.shape[0]
        )
        left_frames = left_frames[:min_frames]
        left_mesh = left_mesh[:min_frames]
        right_frames = right_frames[:min_frames]
        right_mesh = right_mesh[:min_frames]
        front_frames = front_frames[:min_frames]
        front_mesh = front_mesh[:min_frames]

        logger.warning(
            f"Aligned all views to {min_frames} frames based on the shortest video."
        )
    else:
        min_frames = left_frames.shape[0]

    all_mesh_3d = []

    # * 逐帧处理三视点数据，三角测量 3D 网格
    for i in range(min_frames):

        # ! debug
        # if i > 50:
        #     break

        f_mesh = front_mesh[i]
        l_mesh = left_mesh[i]
        r_mesh = right_mesh[i]

        f_frame = front_frames[i]
        l_frame = left_frames[i]
        r_frame = right_frames[i]

        mesh_3d, (target_h, target_w) = process_one_frame(
            f_frame,
            f_mesh,
            l_frame,
            l_mesh,
            r_frame,
            r_mesh,
            rt_info,
            K,
            output_path,
            i,
            vis,
        )

        if mesh_3d is False:
            all_mesh_3d.append(np.full_like(f_mesh, np.nan))
        else:
            all_mesh_3d.append(mesh_3d)

    all_mesh_3d = np.array(all_mesh_3d)  # (T, N, 3)

    # * 检查缺帧 & 自动插值
    all_mesh_3d, info = interpolate_missing_frames(
        all_mesh_3d, method="auto", zero_as_missing=True, zero_axes=(0, 1, 2)
    )
    logger.info(f"Missing frame ratio: {info['miss_ratio']:.3f}")
    logger.info(f"Longest missing run: {info['longest_missing_run']} frames")
    logger.info(f"Interpolation method used: {info['method_effective']}")

    # * 评估原始三角测量结果
    metrics_before = evaluate_face3d_pro(all_mesh_3d, fps=30)
    logger.info("====== Evaluation Metrics Before Smoothing ======")
    for k, v in metrics_before.items():
        logger.info(f"{k}: {v:.5f}")
    # * 导出评估报告
    export_report(
        metrics_before,
        outdir=output_path / "evaluation_report_before_smoothing",
    )

    # * 平滑 3D 网格序列
    all_mesh_3d_smooth, report = smooth_combo(all_mesh_3d, alpha=0.3, win=9, poly=2)

    # * 打印平滑报告
    logger.info("====== Smooth Report ======")
    logger.info(f"Frames: {report['frames']}, Joints: {report['joints']}")
    logger.info(f"Motion Energy Before: {report['motion_energy_before']:.5f}")
    logger.info(f"Motion Energy After : {report['motion_energy_after']:.5f}")
    logger.info(f"Reduction Rate      : {report['reduction_rate']*100:.2f}% ↓")

    # * 单点轨迹可视化
    # TODO:

    # * 全剧平均改变量
    diff = np.nanmean(np.linalg.norm(all_mesh_3d_smooth - all_mesh_3d, axis=2))
    logger.info(f"Average 3D point change after smoothing: {diff:.5f}")

    # * 评估模型输出稳定性和多视一致性
    metrics = evaluate_face3d_pro(all_mesh_3d_smooth, fps=30)

    logger.info("====== Evaluation Metrics ======")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.5f}")

    # * 导出评估报告
    export_report(
        metrics,
        outdir=output_path / "evaluation_report",
    )

    # * 保存 3D 关键点
    for frame_idx in range(all_mesh_3d_smooth.shape[0]):

        if vis.save_mesh_3d and not np.isnan(all_mesh_3d_smooth[frame_idx]).all():

            save_3d_joints(
                mesh_3d=all_mesh_3d_smooth[frame_idx],
                save_dir=output_path / "3d_joints" / "smooth",
                frame_idx=frame_idx,
                rt_info=rt_info,
                k=K,
                video_path={
                    "front": str(environment_dir["front"]["video"]),
                    "left": str(environment_dir["left"]["video"]),
                    "right": str(environment_dir["right"]["video"]),
                },
                npz_path={
                    "front": str(environment_dir["front"]["npz"]),
                    "left": str(environment_dir["left"]["npz"]),
                    "right": str(environment_dir["right"]["npz"]),
                },
            )

            visualize_3d_mesh(
                mesh_3d=all_mesh_3d_smooth[frame_idx],
                save_path=output_path
                / "vis"
                / f"mesh_3D_frames_depth_colored/frame_{frame_idx:04d}.png",
                title=f"Frame {frame_idx} - 3D Mesh (Depth Colored)",
                rt_info=rt_info,
                K=K,
                image_size=(target_w, target_h),
                save_views=["default", "front", "left", "right"],  # 需要哪些就写哪些
                default_view=(20.0, -60.0),
                invert_y=False,
            )

    # * merge 3d mesh visualization frames to video
    if vis.merge_3d_frames_to_video:
        merge_frames_to_video(
            frame_dir=output_path / "vis" / "mesh_3D_frames_depth_colored/top",
            output_video_path=output_path / "vis" / (output_path.stem + "_top.mp4"),
            fps=30,
        )

        merge_frames_to_video(
            frame_dir=output_path / "vis" / "mesh_3D_frames_depth_colored/default",
            output_video_path=output_path / "vis" / (output_path.stem + "_default.mp4"),
            fps=30,
        )


# ---- 子进程全局缓存（由 initializer 注入一次）----
_G = {
    "rt_info": None,
    "K": None,
    "vis": None,
    "video_base": None,
    "output_base": None,
}


def _worker_init(
    rt_info, K, vis_flag, video_base: Path, output_base: Path, log_root: Path
):
    # 限制每个子进程里 BLAS / OpenCV 的线程数，避免过度并行
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    _G["rt_info"] = rt_info
    _G["K"] = K
    _G["vis"] = vis_flag
    _G["video_base"] = Path(video_base)
    _G["output_base"] = Path(output_base)

    _setup_worker_logger(Path(log_root))


def _run_env_task(task):
    """
    task: (env_dir: Path, person_name: str)
    使用全局 _G 里的 rt_info/K/vis/video/output。
    """
    env_dir, person_name = task
    env_dir = Path(env_dir)
    env_name = env_dir.name

    try:
        logger.info(f"[TASK] {person_name}/{env_name} start")
        npz_files = sorted(env_dir.glob("*/*.npz"))
        if not npz_files:
            return (person_name, env_name, False, "No .npz files")

        # 映射 front/left/right
        mapped_info = {}
        for f in npz_files:
            stem = f.stem  # 约定 stem == front/left/right
            mapped_info[stem] = {
                "npz": f,
                "video": _G["video_base"] / person_name / env_name / f"{stem}.mp4",
            }

        out_dir = _G["output_base"] / person_name / env_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 你的原函数：使用 _G 中的参数
        process_one_video(mapped_info, out_dir, _G["rt_info"], _G["K"], _G["vis"])
        logger.info(f"[TASK] {person_name}/{env_name} finish")
        return (person_name, env_name, True, "OK")
    except Exception as e:
        tb = traceback.format_exc(limit=10)
        return (person_name, env_name, False, f"{e}\n{tb}")


# ---------- 多人批量处理入口 ----------
def process_person_videos(
    mesh_path: Path,
    video_path: Path,
    output_path: Path,
    rt_info,
    K,
    vis_flag,
    debug: bool,
):
    subjects = sorted(Path(mesh_path).glob("*/"))
    if not subjects:
        raise FileNotFoundError(f"No folders found in: {mesh_path}")
    logger.info(f"Found {len(subjects)} subjects in {mesh_path}")

    # 收集所有 environment 任务（跨所有 person）
    tasks = []
    for person_dir in subjects:
        person_name = person_dir.name
        logger.info(f"Processing: {person_name}")
        envs = sorted(person_dir.glob("*/"))
        if not envs:
            logger.warning(f"[WARN] No environments found for {person_name}")
            continue
        for env_dir in envs:
            tasks.append((env_dir, person_name))

    if not tasks:
        logger.warning("[WARN] No environment tasks found.")
        return

    if debug:

        logger.debug(f"[DEBUG] Debug mode is enabled.")
        # —— 调试：单进程、第一条任务、可打断点 ——
        env_dir, person_name = tasks[0]
        logger.debug(f"[DEBUG] person={person_name} env={env_dir.name}")
        _worker_init(rt_info, K, vis_flag, Path(video_path), Path(output_path))

        env_dir = Path(env_dir)
        env_name = env_dir.name

        logger.info(f"[TASK] {person_name}/{env_name}")

        npz_files = sorted(env_dir.glob("*/*.npz"))

        if not npz_files:
            return (person_name, env_name, False, "No .npz files")

        # 映射 front/left/right
        mapped_info = {}
        for f in npz_files:
            stem = f.stem  # 约定 stem == front/left/right
            mapped_info[stem] = {
                "npz": f,
                "video": _G["video_base"] / person_name / env_name / f"{stem}.mp4",
            }

        out_dir = _G["output_base"] / person_name / env_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 你的原函数：使用 _G 中的参数
        process_one_video(mapped_info, out_dir, _G["rt_info"], _G["K"], _G["vis"])

    # —— 正式：多进程并行 ——
    else:
        logger.info(f"[INFO] Starting multiprocessing with {len(tasks)} tasks.")
        # 并发度：默认 CPU 一半；IO/SSD 强时可适当调高
        processes = max(1, (os.cpu_count() or 4))
        chunksize = 2  # 小任务可调大，减少调度开销
        maxtasksperchild = 40  # 长跑更稳，防内存涨

        logger.info(
            f"[POOL] tasks={len(tasks)} procs={processes} chunksize={chunksize}"
        )

        log_root = Path(output_path) / "_logs"  # 总日志目录
        ctx = mp.get_context("spawn")  # 跨平台稳
        with ctx.Pool(
            processes=processes,
            initializer=_worker_init,
            initargs=(
                rt_info,
                K,
                vis_flag,
                Path(video_path),
                Path(output_path),
                log_root,
            ),
            maxtasksperchild=maxtasksperchild,
        ) as pool:

            for person, env, ok, msg in pool.imap_unordered(
                _run_env_task, tasks, chunksize
            ):
                if ok:
                    logger.info(f"[OK]   {person}/{env} -> {msg}")
                else:
                    logger.error(f"[FAIL] {person}/{env} -> {msg}")


@hydra.main(
    version_base=None, config_path="../configs", config_name="mesh_triangulation"
)
def main(cfg):

    # 准备相机外部参数
    camera_position_dict = prepare_camera_position(
        K=cfg.camera_K,
        T=cfg.camera_position.T,
        z=cfg.camera_position.z,
        output_path=cfg.paths.log_path,
        img_size=cfg.camera_K.image_size,
        dist_front=cfg.camera_position.dist_front,
        dist_left=cfg.camera_position.dist_left,
        dist_right=cfg.camera_position.dist_right,
        baseline=cfg.camera_position.baseline,
    )

    process_person_videos(
        mesh_path=Path(cfg.paths.mesh_path),
        video_path=Path(cfg.paths.video_path),
        output_path=Path(cfg.paths.log_path),
        rt_info=camera_position_dict["rt_info"],
        K=camera_position_dict["K_map"],
        vis_flag=cfg.vis,
        debug=cfg.debug,
    )


if __name__ == "__main__":

    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()

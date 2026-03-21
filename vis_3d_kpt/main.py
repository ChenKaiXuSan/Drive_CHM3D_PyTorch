#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import time
import resource
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from vis_3d_kpt.load import OnePersonInfo, np
from vis_3d_kpt.visualize import run_visualization

logger = logging.getLogger(__name__)


def _configure_logging(log_dir: Path) -> tuple[Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    all_log_path = log_dir / f"run_{run_id}.all.log"
    error_log_path = log_dir / f"run_{run_id}.error.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    log_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(log_format)

    all_file_handler = logging.FileHandler(all_log_path, encoding="utf-8")
    all_file_handler.setLevel(logging.INFO)
    all_file_handler.setFormatter(log_format)

    error_file_handler = logging.FileHandler(error_log_path, encoding="utf-8")
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(log_format)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(all_file_handler)
    root_logger.addHandler(error_file_handler)

    return all_log_path, error_log_path


def _get_peak_rss_mb() -> float:
    # Linux ru_maxrss 单位是 KB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _load_frame_keypoints(frame_path: Path) -> np.ndarray:
    """加载单帧关键点，兼容数组与 dict 封装的 npy。"""
    loaded = np.load(frame_path, allow_pickle=True)

    # 某些文件保存为 object 标量，内部是 dict
    if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
        loaded = loaded.item()

    if isinstance(loaded, dict):
        if "fused_keypoints_3d" not in loaded:
            raise KeyError(f"{frame_path} 缺少 fused_keypoints_3d 字段")
        loaded = loaded["fused_keypoints_3d"]

    return np.asarray(loaded, dtype=np.float32)


def _build_sequence_npy(frame_files: list[Path], save_path: Path) -> Path:
    """将按帧保存的 npy 文件流式写为 (T, N, 3) 序列并落盘。"""
    if not frame_files:
        raise ValueError("frame_files 为空，无法构建序列")

    t0 = time.perf_counter()
    logger.info(
        "开始构建序列: %s, 帧数=%d",
        str(save_path),
        len(frame_files),
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)

    first = _load_frame_keypoints(frame_files[0])
    seq_shape = (len(frame_files),) + first.shape
    seq_dtype = np.float32
    seq_mmap = np.lib.format.open_memmap(
        str(save_path),
        mode="w+",
        dtype=seq_dtype,
        shape=seq_shape,
    )
    seq_mmap[0] = np.asarray(first, dtype=seq_dtype)

    for idx, fp in enumerate(frame_files[1:], start=1):
        seq_mmap[idx] = np.asarray(_load_frame_keypoints(fp), dtype=seq_dtype)
        if idx % 300 == 0:
            logger.info(
                "序列构建进度: %s %d/%d",
                save_path.name,
                idx + 1,
                len(frame_files),
            )

    seq_mmap.flush()
    del seq_mmap
    logger.info(
        "序列构建完成: %s, shape=%s, 耗时=%.2fs, 进程峰值内存=%.1fMB",
        str(save_path),
        seq_shape,
        time.perf_counter() - t0,
        _get_peak_rss_mb(),
    )
    return save_path


def _resolve_required_paths(
    args: argparse.Namespace,
    person_id: str,
    env_name: str,
) -> tuple[Path, Path, Path, Path, Path, Path]:
    """解析单个 person/env 需要的视频与2D关键点路径。"""
    video_dir = args.video_dir.resolve()
    sam3d_results_dir = args.sam3d_results_dir.resolve()

    left_video = video_dir / person_id / env_name / "left.mp4"
    right_video = video_dir / person_id / env_name / "right.mp4"
    front_video = video_dir / person_id / env_name / "front.mp4"

    left_2d = sam3d_results_dir / person_id / env_name / "left"
    right_2d = sam3d_results_dir / person_id / env_name / "right"
    front_2d = sam3d_results_dir / person_id / env_name / "front"

    return left_video, right_video, front_video, left_2d, right_2d, front_2d


def _all_paths_exist(paths: list[Path]) -> bool:
    for p in paths:
        if not p.exists():
            logger.warning("缺少输入路径，跳过: %s", p)
            return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量可视化 head3d_fuse_results 中的 3D 融合关键点。"
    )
    parser.add_argument(
        "--head3d-dir",
        type=Path,
        default=Path("/workspace/data/head3d_fuse_results"),
        help="head3d_fuse_results 根目录，结构为 {person_id}/{env_name}/{fused|smoothed}_npz/",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("/workspace/data/videos_split"),
        help="视频文件所在目录（可选），用于叠加原始视频",
    )
    parser.add_argument(
        "--sam3d-results-dir",
        type=Path,
        default=Path("/workspace/data/sam3d_body_results_right_full"),
        help="sam3d 2D 关键点结果目录（可选）",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/data/fused_3d_vis"),
        help="输出目录，将保留原目录结构",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/workspace/code/logs/vis_3d_kpt"),
        help="日志目录，会生成按时间戳命名的 all/error 日志文件",
    )
    parser.add_argument(
        "--person-list",
        type=str,
        default=None,
        help="要处理的 person ID 列表，逗号分隔（如 01,02,03），默认处理所有",
    )
    parser.add_argument(
        "--env-list",
        type=str,
        default=None,
        help="要处理的环境名称列表，逗号分隔，默认处理所有",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=["fused", "smoothed", "both"],
        default="both",
        help="处理数据类型：fused（融合）、smoothed（平滑）或 both（两者）",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="并行工作数量，0 表示自动使用全部 CPU 核数",
    )
    parser.add_argument(
        "--executor",
        type=str,
        choices=["thread", "process"],
        default="process",
        help="并行执行器类型：process 更适合 CPU 密集，thread 更适合 I/O 密集",
    )
    parser.add_argument(
        "--pipeline-mode",
        type=str,
        choices=["env", "staged"],
        default="env",
        help="并行流水线模式：env=每个任务串行建序列+渲染，staged=先全量建序列再并行渲染",
    )
    parser.add_argument(
        "--prepare-workers",
        type=int,
        default=8,
        help="staged 模式下建序列阶段并行数，默认 8",
    )
    parser.add_argument(
        "--prepare-executor",
        type=str,
        choices=["thread", "process"],
        default="thread",
        help="staged 模式下建序列执行器类型，默认 thread（通常更适合磁盘 I/O）",
    )
    return parser.parse_args()


def _resolve_num_workers(num_workers: int, task_count: int, cap: int = 16) -> int:
    cpu_count = os.cpu_count() or 1
    if num_workers <= 0:
        num_workers = cpu_count if cap <= 0 else min(cap, cpu_count)
    return max(1, min(num_workers, task_count))


def _prepare_single_env(
    args: argparse.Namespace,
    person_id: str,
    env_name: str,
    env_data: dict,
    out_dir: Path,
) -> tuple[str, str, bool, str, dict]:
    try:
        prep_t0 = time.perf_counter()
        if not env_data["fused"] or not env_data["smoothed"]:
            return person_id, env_name, False, "缺少 fused/smoothed 数据", {}

        (
            left_video_path,
            right_video_path,
            front_video_path,
            left_2d_kpt_path,
            right_2d_kpt_path,
            front_2d_kpt_path,
        ) = _resolve_required_paths(args, person_id, env_name)

        if not _all_paths_exist(
            [
                left_video_path,
                right_video_path,
                front_video_path,
                left_2d_kpt_path,
                right_2d_kpt_path,
                front_2d_kpt_path,
            ]
        ):
            return person_id, env_name, False, "输入路径不完整", {}

        seq_cache_dir = out_dir / person_id / env_name / "_sequence_cache"
        fused_seq_path = _build_sequence_npy(
            env_data["fused"], seq_cache_dir / "fused_sequence.npy"
        )
        smoothed_seq_path = _build_sequence_npy(
            env_data["smoothed"], seq_cache_dir / "smoothed_sequence.npy"
        )

        payload = {
            "left_video_path": left_video_path,
            "right_video_path": right_video_path,
            "front_video_path": front_video_path,
            "left_2d_kpt_path": left_2d_kpt_path,
            "right_2d_kpt_path": right_2d_kpt_path,
            "front_2d_kpt_path": front_2d_kpt_path,
            "fused_seq_path": fused_seq_path,
            "smoothed_seq_path": smoothed_seq_path,
            "prepare_seconds": time.perf_counter() - prep_t0,
        }
        return person_id, env_name, True, "prepare 完成", payload
    except Exception as e:
        return person_id, env_name, False, f"prepare 失败: {str(e)}", {}


def _render_single_env(
    person_id: str,
    env_name: str,
    prepared: dict,
    out_dir: Path,
) -> tuple[str, str, bool, str]:
    try:
        render_t0 = time.perf_counter()
        run_visualization(
            person_info=OnePersonInfo(
                person_name=person_id,
                env_name=env_name,
                left_video_path=prepared["left_video_path"],
                right_video_path=prepared["right_video_path"],
                front_video_path=prepared["front_video_path"],
                left_2d_kpt_path=prepared["left_2d_kpt_path"],
                right_2d_kpt_path=prepared["right_2d_kpt_path"],
                front_2d_kpt_path=prepared["front_2d_kpt_path"],
                fused_3d_kpt_path=prepared["fused_seq_path"],
                fused_smoothed_3d_kpt_path=prepared["smoothed_seq_path"],
            ),
            out_dir=out_dir / person_id / env_name,
        )
        msg = (
            f"✅ {person_id}/{env_name} 完成, "
            f"prepare={prepared.get('prepare_seconds', -1):.2f}s, "
            f"render={time.perf_counter() - render_t0:.2f}s"
        )
        return person_id, env_name, True, msg
    except Exception as e:
        return person_id, env_name, False, f"❌ {person_id}/{env_name} 渲染失败: {str(e)}"


def _process_single_env(
    args: argparse.Namespace,
    person_id: str,
    env_name: str,
    env_data: dict,
    out_dir: Path,
) -> tuple[str, str, bool, str]:
    """处理单个 env，返回 (person_id, env_name, success, message)。"""
    try:
        env_t0 = time.perf_counter()
        logger.info(
            "开始处理 env: %s/%s, fused=%d, smoothed=%d",
            person_id,
            env_name,
            len(env_data.get("fused", [])),
            len(env_data.get("smoothed", [])),
        )
        _, _, prep_ok, prep_msg, prepared = _prepare_single_env(
            args, person_id, env_name, env_data, out_dir
        )
        if not prep_ok:
            return person_id, env_name, False, prep_msg

        _, _, render_ok, render_msg = _render_single_env(
            person_id, env_name, prepared, out_dir
        )
        if not render_ok:
            return person_id, env_name, False, render_msg

        msg = (
            f"✅ {person_id}/{env_name} 完成, "
            f"耗时={time.perf_counter() - env_t0:.2f}s, "
            f"峰值内存={_get_peak_rss_mb():.1f}MB"
        )
        return person_id, env_name, True, msg
    except Exception as e:
        msg = f"❌ {person_id}/{env_name} 失败: {str(e)}"
        return person_id, env_name, False, msg


def find_head3d_fuse_results(
    head3d_dir: Path,
    person_list: list | None = None,
    env_list: list | None = None,
    data_type: str = "both",
) -> dict:
    """在 head3d_fuse_results 目录下按人员和环境组织关键点数据。

    返回结构：
    {
        "person_id": {
            "env_name": {
                "fused": [frame_npy_path1, frame_npy_path2, ...],
                "smoothed": [frame_npy_path1, frame_npy_path2, ...],
            },
            ...
        },
        ...
    }
    """
    result = {}

    if not head3d_dir.exists():
        logger.warning(f"head3d_fuse_results 目录不存在: {head3d_dir}")
        return result

    # 遍历所有人员
    for person_dir in sorted(head3d_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name

        # 过滤人员
        if person_list and person_id not in person_list:
            continue

        result[person_id] = {}

        # 遍历所有环境
        for env_dir in sorted(person_dir.iterdir()):
            if not env_dir.is_dir():
                continue

            env_name = env_dir.name

            # 过滤环境
            if env_list and env_name not in env_list:
                continue

            result[person_id][env_name] = {"fused": [], "smoothed": []}

            # 读取融合数据
            if data_type in ["fused", "both"]:
                fused_dir = env_dir / "fused_npz"
                if fused_dir.exists():
                    fused_files = sorted(fused_dir.glob("*_fused.npy"))
                    result[person_id][env_name]["fused"] = fused_files

            # 读取平滑数据
            if data_type in ["smoothed", "both"]:
                smoothed_dir = env_dir / "smoothed_fused_npz"
                if smoothed_dir.exists():
                    smoothed_files = sorted(smoothed_dir.glob("*_fused.npy"))
                    result[person_id][env_name]["smoothed"] = smoothed_files

    return result


if __name__ == "__main__":
    args = parse_args()
    all_log_path, error_log_path = _configure_logging(args.log_dir.resolve())
    logger.info("日志文件(all): %s", str(all_log_path))
    logger.info("日志文件(error): %s", str(error_log_path))

    main_t0 = time.perf_counter()
    out_dir = args.out_dir.resolve()
    head3d_dir = args.head3d_dir.resolve()

    # 这些目录是必需输入
    args.video_dir = args.video_dir.resolve()
    args.sam3d_results_dir = args.sam3d_results_dir.resolve()

    # 解析人员和环境列表
    person_list = None
    if args.person_list:
        person_list = [p.strip() for p in args.person_list.split(",")]

    env_list = None
    if args.env_list:
        env_list = [e.strip() for e in args.env_list.split(",")]

    # 查找数据
    logger.info(
        "开始扫描数据: head3d=%s, video=%s, sam3d=%s, out=%s",
        str(head3d_dir),
        str(args.video_dir),
        str(args.sam3d_results_dir),
        str(out_dir),
    )
    head3d_results = find_head3d_fuse_results(
        head3d_dir,
        person_list=person_list,
        env_list=env_list,
        data_type=args.data_type,
    )

    if not head3d_results:
        logger.error(f"未在 {head3d_dir} 中找到数据")
        exit(1)

    # 显示找到的数据并收集任务
    tasks = []
    for person_id in sorted(head3d_results.keys()):
        logger.info(f"Person {person_id}:")
        for env_name in sorted(head3d_results[person_id].keys()):
            env_data = head3d_results[person_id][env_name]
            fused_count = len(env_data["fused"])
            smoothed_count = len(env_data["smoothed"])
            logger.info(f"  {env_name}: {fused_count} 融合, {smoothed_count} 平滑")
            tasks.append((person_id, env_name, env_data))

    if not tasks:
        logger.warning("没有找到要处理的 env")
        logger.info(f"输出目录: {out_dir}")
        exit(0)

    completed = 0
    failed = 0

    if args.pipeline_mode == "env":
        worker_count = _resolve_num_workers(args.num_workers, len(tasks), cap=0)
        executor_cls = (
            ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        )
        logger.info(
            "\n开始并行处理 %d 个 env，模式=%s，执行器=%s，工作数=%d\n",
            len(tasks),
            args.pipeline_mode,
            args.executor,
            worker_count,
        )
        with executor_cls(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    _process_single_env, args, person_id, env_name, env_data, out_dir
                ): (person_id, env_name)
                for person_id, env_name, env_data in tasks
            }

            for future in as_completed(futures):
                person_id, env_name, success, message = future.result()
                if success:
                    logger.info(message)
                    completed += 1
                else:
                    logger.warning(message)
                    failed += 1
    else:
        prepare_t0 = time.perf_counter()
        prepare_worker_count = _resolve_num_workers(
            args.prepare_workers, len(tasks), cap=8
        )
        prepare_executor_cls = (
            ProcessPoolExecutor
            if args.prepare_executor == "process"
            else ThreadPoolExecutor
        )
        logger.info(
            "\n阶段1/2 prepare：任务=%d，执行器=%s，工作数=%d\n",
            len(tasks),
            args.prepare_executor,
            prepare_worker_count,
        )

        prepared_items = []
        with prepare_executor_cls(max_workers=prepare_worker_count) as executor:
            futures = {
                executor.submit(
                    _prepare_single_env, args, person_id, env_name, env_data, out_dir
                ): (person_id, env_name)
                for person_id, env_name, env_data in tasks
            }
            for future in as_completed(futures):
                person_id, env_name, success, message, payload = future.result()
                if success:
                    prepared_items.append((person_id, env_name, payload))
                    logger.info("[PREP] %s/%s %s", person_id, env_name, message)
                else:
                    failed += 1
                    logger.warning("[PREP] %s/%s %s", person_id, env_name, message)

        prepare_seconds = time.perf_counter() - prepare_t0
        if not prepared_items:
            logger.error("prepare 阶段没有成功任务，提前结束")
            logger.info(f"输出目录: {out_dir}")
            exit(1)

        render_t0 = time.perf_counter()
        render_worker_count = _resolve_num_workers(
            args.num_workers, len(prepared_items), cap=0
        )
        render_executor_cls = (
            ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
        )
        logger.info(
            "\n阶段2/2 render：任务=%d，执行器=%s，工作数=%d\n",
            len(prepared_items),
            args.executor,
            render_worker_count,
        )
        with render_executor_cls(max_workers=render_worker_count) as executor:
            futures = {
                executor.submit(
                    _render_single_env, person_id, env_name, prepared, out_dir
                ): (person_id, env_name)
                for person_id, env_name, prepared in prepared_items
            }
            for future in as_completed(futures):
                person_id, env_name, success, message = future.result()
                if success:
                    completed += 1
                    logger.info("[RENDER] %s", message)
                else:
                    failed += 1
                    logger.warning("[RENDER] %s", message)

        render_seconds = time.perf_counter() - render_t0
        logger.info(
            "\n流水线耗时统计: prepare=%.2fs, render=%.2fs, prepare吞吐=%.2f env/s, render吞吐=%.2f env/s",
            prepare_seconds,
            render_seconds,
            len(tasks) / max(prepare_seconds, 1e-6),
            len(prepared_items) / max(render_seconds, 1e-6),
        )

    logger.info(f"\n✅ 完成: {completed} 成功, {failed} 失败")
    logger.info(f"输出目录: {out_dir}")
    logger.info(
        "总耗时=%.2fs, 进程峰值内存=%.1fMB",
        time.perf_counter() - main_t0,
        _get_peak_rss_mb(),
    )

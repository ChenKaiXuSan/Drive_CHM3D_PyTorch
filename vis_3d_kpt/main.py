#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vis_3d_kpt.load import OnePersonInfo, np
from vis_3d_kpt.visualize import run_visualization

logger = logging.getLogger(__name__)


def _build_sequence_npy(frame_files: list[Path], save_path: Path) -> Path:
    """将按帧保存的 npy 文件堆叠为 (T, N, 3) 序列并落盘。"""
    if not frame_files:
        raise ValueError("frame_files 为空，无法构建序列")

    frames = [np.asarray(np.load(fp, allow_pickle=True), dtype=np.float32) for fp in frame_files]
    seq = np.stack(frames, axis=0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, seq)
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
        default=Path("/workspace/code/logs/3d_vis_head3d"),
        help="输出目录，将保留原目录结构",
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
    return parser.parse_args()


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
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
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
    head3d_results = find_head3d_fuse_results(
        head3d_dir,
        person_list=person_list,
        env_list=env_list,
        data_type=args.data_type,
    )

    if not head3d_results:
        logger.error(f"未在 {head3d_dir} 中找到数据")
        exit(1)

    # 显示找到的数据
    for person_id in sorted(head3d_results.keys()):
        logger.info(f"Person {person_id}:")
        for env_name in sorted(head3d_results[person_id].keys()):
            env_data = head3d_results[person_id][env_name]
            fused_count = len(env_data["fused"])
            smoothed_count = len(env_data["smoothed"])
            logger.info(f"  {env_name}: {fused_count} 融合, {smoothed_count} 平滑")

            if not env_data["fused"] or not env_data["smoothed"]:
                logger.warning("缺少 fused/smoothed 数据，跳过 %s/%s", person_id, env_name)
                continue

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
                continue

            # 输入是按帧 npy，先堆叠成序列 npy 再交给 run_visualization
            seq_cache_dir = out_dir / person_id / env_name / "_sequence_cache"
            fused_seq_path = _build_sequence_npy(
                env_data["fused"], seq_cache_dir / "fused_sequence.npy"
            )
            smoothed_seq_path = _build_sequence_npy(
                env_data["smoothed"], seq_cache_dir / "smoothed_sequence.npy"
            )

            run_visualization(
                person_info=OnePersonInfo(
                    person_name=person_id,
                    env_name=env_name,
                    left_video_path=left_video_path,
                    right_video_path=right_video_path,
                    front_video_path=front_video_path,
                    left_2d_kpt_path=left_2d_kpt_path,
                    right_2d_kpt_path=right_2d_kpt_path,
                    front_2d_kpt_path=front_2d_kpt_path,
                    fused_3d_kpt_path=fused_seq_path,
                    fused_smoothed_3d_kpt_path=smoothed_seq_path,
                ),
                out_dir=out_dir / person_id / env_name,
            )

    logger.info("✅ 所有可视化已完成！")
    logger.info(f"输出目录: {out_dir}")

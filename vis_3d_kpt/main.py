#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

from .load import OnePersonInfo
from .visualize import run_visualization

logger = logging.getLogger(__name__)


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
        "--out-dir",
        type=Path,
        default=Path("/workspace/code/logs/3d_vis_head3d"),
        help="输出目录，将保留原目录结构",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="视频文件所在目录（可选），用于叠加原始视频",
    )
    parser.add_argument(
        "--sam3d-results-dir",
        type=Path,
        default=None,
        help="sam3d 2D 关键点结果目录（可选）",
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
                    smoothed_files = sorted(smoothed_dir.glob("*_smoothed.npy"))
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

            run_visualization(
                person_info=OnePersonInfo(
                    person_name=person_id,
                    left_video_path=video_dir / person_id / env_name / "left.mp4"
                    if video_dir
                    else Path(""),
                    right_video_path=video_dir / person_id / env_name / "right.mp4"
                    if video_dir
                    else Path(""),
                    left_2d_kpt_path=sam3d_results_dir
                    / person_id
                    / env_name
                    / "left.npz"
                    if sam3d_results_dir
                    else Path(""),
                    right_2d_kpt_path=sam3d_results_dir
                    / person_id
                    / env_name
                    / "right.npz"
                    if sam3d_results_dir
                    else Path(""),
                    fused_3d_kpt_path=env_data["fused"][0]
                    if env_data["fused"]
                    else Path(""),
                    fused_smoothed_3d_kpt_path=env_data["smoothed"][0]
                    if env_data["smoothed"]
                    else Path(""),
                ),
                output_dir=out_dir / person_id / env_name,
            )

    logger.info("✅ 所有可视化已完成！")
    logger.info(f"输出目录: {out_dir}")

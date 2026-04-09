#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
与人工标注比较的配置和默认路径
"""

from pathlib import Path

# =============== 默认路径配置 ===============

# 数据路径
DATA_ROOT = Path("/workspace/data")

# 融合关键点数据路径
FUSED_NPZ_ROOT = DATA_ROOT / "head3d_fuse_results_old"

# 单视角 SAM3D 结果路径
SAM3D_BODY_RESULTS_ROOT = DATA_ROOT / "sam3d_body_results_right_full"

# 融合关键点数据子目录
FUSED_SUBDIR_RAW = "fused_npz"
FUSED_SUBDIR_SMOOTHED = "smoothed_fused_npz"

# 标注文件路径
ANNOTATION_FILE = DATA_ROOT / "annotation" / "label" / "full.json"

# 输出结果路径
OUTPUT_ROOT = DATA_ROOT / "compare_with_human_annotation_results"

# 输出结果的来源分层目录
OUTPUT_ROOT_RAW = OUTPUT_ROOT / "fused" / "raw"
OUTPUT_ROOT_SMOOTHED = OUTPUT_ROOT / "fused" / "smoothed"

# 多数投票后的输出结果路径（在OUTPUT_ROOT下）
OUTPUT_ROOT_MAJORITY_VOTE = OUTPUT_ROOT_RAW / "majority"

# 按标注者分别比较的输出结果路径（在OUTPUT_ROOT下）
OUTPUT_ROOT_BY_ANNOTATOR = OUTPUT_ROOT_RAW / "by_annotator"

# 单视角 SAM3D 输出路径
OUTPUT_ROOT_SAM3D_VIEWS = OUTPUT_ROOT / "sam3d_views"

# =============== 快速配置 ===============

# 默认分析的 person 和 environment
DEFAULT_PERSON_ID = "01"
DEFAULT_ENV_JP = "昼多い"  # 或 "夜多い", "昼少ない", "夜少ない"

# 默认的 frame 范围
DEFAULT_START_FRAME = None  # None 表示从头开始
DEFAULT_END_FRAME = None  # None 表示到结尾

# 默认的匹配阈值（度）
DEFAULT_THRESHOLD = 5.0

# 支持的环境
ENVIRONMENTS = {
    "昼多い": "day_high",
    "昼少ない": "day_low",
    "夜多い": "night_high",
    "夜少ない": "night_low",
}

# 支持的单视角名称
SAM3D_VIEWS = ("front", "left", "right")

# =============== 便利函数 ===============


def get_fused_dir(person_id: str, env_jp: str, smoothed: bool = False) -> Path:
    """获取融合数据目录。"""
    fused_subdir = FUSED_SUBDIR_SMOOTHED if smoothed else FUSED_SUBDIR_RAW
    return FUSED_NPZ_ROOT / person_id / env_jp / fused_subdir


def get_sam3d_view_dir(person_id: str, env_jp: str, view: str) -> Path:
    """获取单视角 SAM3D 结果目录。"""
    return SAM3D_BODY_RESULTS_ROOT / person_id / env_jp / view


def get_output_source_root(smoothed: bool = False) -> Path:
    """获取按输入来源分层的输出根目录。"""
    return OUTPUT_ROOT_SMOOTHED if smoothed else OUTPUT_ROOT_RAW


def get_sam3d_output_dir(
    person_id: str,
    env_en: str,
    view: str,
    output_root: Path = OUTPUT_ROOT_SAM3D_VIEWS,
) -> Path:
    """创建并获取单视角 SAM3D 的输出目录。"""
    output_dir = output_root / person_id / env_en / view
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_annotation_file() -> Path:
    """获取标注文件路径"""
    return ANNOTATION_FILE


def get_output_dir(
    person_id: str,
    env_en: str,
    output_root: Path = OUTPUT_ROOT,
) -> Path:
    """创建和获取输出目录"""
    output_dir = output_root / person_id / env_en
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    print("=" * 60)
    print("默认路径配置")
    print("=" * 60)
    print(f"\n数据根目录: {DATA_ROOT}")
    print(f"融合NPZ目录: {FUSED_NPZ_ROOT}")
    print(f"标注文件: {ANNOTATION_FILE}")
    print(f"输出目录: {OUTPUT_ROOT}")

    print("\n默认配置:")
    print(f"  Person ID: {DEFAULT_PERSON_ID}")
    print(f"  Environment: {DEFAULT_ENV_JP}")
    print(f"  Frame范围: {DEFAULT_START_FRAME} ~ {DEFAULT_END_FRAME}")
    print(f"  匹配阈值: {DEFAULT_THRESHOLD}°")

    # 检查文件是否存在
    print("\n文件检查:")
    annotation_exists = get_annotation_file().exists()
    print(f"  标注文件: {'✓' if annotation_exists else '✗'} {get_annotation_file()}")

    fused_dir = get_fused_dir(DEFAULT_PERSON_ID, DEFAULT_ENV_JP)
    fused_exists = fused_dir.exists()
    print(f"  融合数据: {'✓' if fused_exists else '✗'} {fused_dir}")

    print("\n" + "=" * 60)

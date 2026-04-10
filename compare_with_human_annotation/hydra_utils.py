#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Hydra 配置加载与路径辅助函数。"""

from pathlib import Path
from typing import Iterable, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def load_config(overrides: Optional[Iterable[str]] = None) -> DictConfig:
    """从 compare_with_human_annotation/conf 加载 Hydra 配置。"""
    config_dir = Path(__file__).resolve().parent / "conf"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="config", overrides=list(overrides or []))


def _as_path(value) -> Path:
    return Path(str(value))


def format_threshold_folder(threshold_deg: float) -> str:
    """格式化阈值目录名，例如 threshold_5deg 或 threshold_7.5deg。"""
    return f"threshold_{threshold_deg:g}deg"


def get_annotation_file(cfg: DictConfig) -> Path:
    return _as_path(cfg.paths.annotation_file)


def get_fused_root(cfg: DictConfig) -> Path:
    return _as_path(cfg.paths.fused_npz_root)


def get_sam3d_root(cfg: DictConfig) -> Path:
    return _as_path(cfg.paths.sam3d_body_results_root)


def get_output_root(cfg: DictConfig, threshold_deg: Optional[float] = None) -> Path:
    root = _as_path(cfg.paths.output_root)
    if threshold_deg is None:
        return root
    return root / format_threshold_folder(float(threshold_deg))


def get_single_view_output_root(cfg: DictConfig, threshold_deg: Optional[float] = None) -> Path:
    return get_output_root(cfg, threshold_deg) / "single_view"


def get_fused_dir(
    cfg: DictConfig,
    person_id: str,
    env_jp: str,
    smoothed: bool = False,
) -> Path:
    fused_subdir = cfg.fused.subdir_smoothed if smoothed else cfg.fused.subdir_raw
    return get_fused_root(cfg) / person_id / env_jp / str(fused_subdir)


def get_sam3d_view_dir(cfg: DictConfig, person_id: str, env_jp: str, view: str) -> Path:
    return get_sam3d_root(cfg) / person_id / env_jp / view


def get_output_source_root(
    cfg: DictConfig,
    smoothed: bool = False,
    threshold_deg: Optional[float] = None,
) -> Path:
    return get_output_root(cfg, threshold_deg) / "fused" / (
        "smoothed" if smoothed else "raw"
    )


def get_sam3d_output_dir(
    cfg: DictConfig,
    person_id: str,
    env_en: str,
    view: str,
    output_root: Optional[Path] = None,
) -> Path:
    root = output_root or get_single_view_output_root(cfg)
    output_dir = root / person_id / env_en / view
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_dir(
    person_id: str,
    env_en: str,
    output_root: Path,
) -> Path:
    output_dir = output_root / person_id / env_en
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_env_mapping(cfg: DictConfig) -> dict:
    return {str(k): str(v) for k, v in cfg.environments.items()}


def get_sam3d_views(cfg: DictConfig) -> tuple:
    return tuple(str(v) for v in cfg.sam3d.views)

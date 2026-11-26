#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/sam3d_body/save.py
Project: /workspace/code/sam3d_body
Created Date: Friday December 5th 2025
Author: Kaixu Chen
-----
Comment:
因为数据太多了，所以按照一帧一个 npz 文件来存储

Save utilities for SAM-3D-Body
- ONLY supports per-frame saving
- One frame -> one .npz

Have a good code time :)
-----
Last Modified: Friday December 5th 2025 11:52:16 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_object_array(x: Any) -> np.ndarray:
    """
    Wrap arbitrary python object (dict with numpy arrays etc.)
    into numpy object array for np.savez.
    """
    return np.array(x, dtype=object)


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def get_frame_path(save_dir: Path, frame_idx: int) -> Path:
    return save_dir / f"{frame_idx:06d}_sam3d_body.npz"


def save_frame(
    output: Dict[str, Any],
    save_dir: Path,
    frame_idx: int,
) -> Path:
    """
    Save ONE frame result.

    Args:
        output:
            dict for one frame (can include numpy arrays, lists, etc.)
        save_dir:
            directory to save into
        frame_idx:
            frame index (used in filename)

    Returns:
        Path to saved .npz
    """
    if not isinstance(output, dict):
        raise TypeError(f"output must be Dict[str, Any], got {type(output)}")

    _ensure_dir(save_dir)

    save_path = get_frame_path(save_dir, frame_idx)

    # 先写临时文件再改名，中断时不会留下半个文件
    tmp_path = save_dir / f".{frame_idx:06d}.{os.getpid()}.tmp"
    try:
        with tmp_path.open("wb") as f:
            np.savez_compressed(f, output=_to_object_array(output))
        os.replace(tmp_path, save_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    logger.info(f"[SAVE] frame {frame_idx} -> {save_path}")
    return save_path

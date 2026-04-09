#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/save.py
Project: /workspace/code/triangulation
Created Date: Friday October 10th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday October 10th 2025 10:58:39 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""
from typing import Dict
from pathlib import Path
import logging
import os
import tempfile
import numpy as np

logger = logging.getLogger(__name__)


def _save_fused_keypoints(
    save_dir: Path,
    frame_idx: int,
    fused_keypoints: np.ndarray,
    fused_mask: np.ndarray,
    n_valid: np.ndarray,
    npz_paths: Dict[str, Path],
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"frame_{frame_idx:06d}_fused.npy"
    payload = {
        "frame_idx": frame_idx,
        "fused_keypoints_3d": fused_keypoints,
        "fused_mask": fused_mask,
        "valid_views": n_valid,
        "npz_paths": {view: str(path) for view, path in npz_paths.items()},
    }

    # Atomic write: write to a temp file first, then replace target file.
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=save_dir, prefix=f".{save_path.stem}.", suffix=".tmp", delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            np.save(tmp_file, payload)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())

        os.replace(tmp_path, save_path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()

    return save_path
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
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


def save_3d_joints(
    mesh_3d: np.ndarray,
    save_dir: Path,
    frame_idx: int,
    rt_info: dict[str, dict[str, np.ndarray]],
    k: dict[int, np.ndarray],
    video_path: dict[str, str],
    npz_path: dict[str, str],
):
    """
    保存3D关节坐标到文件（支持 npy / csv / json）

    Args:
        joints_3d (np.ndarray): (J,3) 关节坐标，单位可为m或任意世界单位
        save_dir (str): 输出文件夹路径
        frame_idx (int): 当前帧编号
        fmt (str): 保存格式，可选 ['npy', 'csv', 'json']
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    fname_base = f"frame_{frame_idx:04d}"

    save_info = {
        "frame": frame_idx,
        "num_joints": len(mesh_3d),
        "joints_3d": mesh_3d,
        "rt_info": rt_info,  # 保存所有相机的RT信息
        "K": k,
        "video_path": video_path,
        "npz_path": npz_path,
    }
    np.save(str(save_dir / f"{fname_base}.npy"), save_info)

    logger.info(
        f"3D joints and info saved (npy) → {str(save_dir / f'{fname_base}.npy')}"
    )

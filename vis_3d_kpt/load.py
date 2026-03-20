#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vis_3d_kpt/loag.py
Project: /workspace/code/vis_3d_kpt
Created Date: Friday March 13th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday March 13th 2026 5:09:45 pm
Modified By: the developer formerly known as Kaixu Chen at
<chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np


@dataclass
class OnePersonInfo:
    person_name: str = ""
    env_name: str = ""
    # 视频路径
    left_video_path: Path = Path("")
    right_video_path: Path = Path("")
    front_video_path: Path = Path("")
    
    # 2d keypoints
    left_2d_kpt_path: Path = Path("")
    right_2d_kpt_path: Path = Path("")
    front_2d_kpt_path: Path = Path("")

    # 融合之后的3d kpt路径
    fused_3d_kpt_path: Path = Path("")
    fused_smoothed_3d_kpt_path: Path = Path("")


def load_fused_3d_kpt(
    file_path: Path,
    use_memmap: bool = False,
) -> np.ndarray:
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if use_memmap:
        # 使用内存映射，节省内存
        loaded = np.load(file_path, mmap_mode='r', allow_pickle=True)
    else:
        loaded = np.load(file_path, allow_pickle=True)

    return loaded


def load_2d_keypoints(file_path: Path, as_generator: bool = False):
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    if file_path.is_file() and file_path.suffix in [".npz"]:
        loaded = np.load(file_path, allow_pickle=True)["outputs"]

        if as_generator:
            # 返回生成器，逐帧yield
            for i in loaded:
                if isinstance(i, dict) and "pred_keypoints_2d" in i:
                    yield i["pred_keypoints_2d"]
        else:
            res_list = []
            for i in loaded:
                if isinstance(i, dict) and "pred_keypoints_2d" in i:
                    kpts_2d = i["pred_keypoints_2d"]
                    res_list.append(kpts_2d)
            return np.asarray(res_list, dtype=np.float64)

    elif file_path.is_dir():
        npz_files = sorted(list(file_path.rglob("*.npz")))
        if not npz_files:
            raise FileNotFoundError(f"目录 {file_path} 中未找到 .npz 文件")

        if as_generator:
            # 返回生成器，逐文件yield
            for npz_file in npz_files:
                loaded = np.load(npz_file, allow_pickle=True)
                yield loaded["outputs"][0]["pred_keypoints_2d"]
        else:
            res_list = []
            for npz_file in npz_files:
                loaded = np.load(npz_file, allow_pickle=True)
                kpts_2d = loaded["outputs"][0]["pred_keypoints_2d"]
                res_list.append(kpts_2d)
            return np.asarray(res_list, dtype=np.float64)

    else:
        raise ValueError(f"无法处理的文件路径: {file_path}")

    # 清理内存
    if 'loaded' in locals():
        del loaded


def load_video_frames(video_path: Path, as_generator: bool = False):
    if not video_path.exists():
        raise FileNotFoundError(f"文件不存在: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if as_generator:
        # 返回生成器，逐帧yield，节省内存
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame_rgb
        finally:
            cap.release()
    else:
        # 原有行为，一次性加载所有帧
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return np.asarray(frames, dtype=np.uint8)


def load_helper(
    person_info: OnePersonInfo,
    memory_efficient: bool = False,
) -> Tuple[
    Union[np.ndarray, list], Union[np.ndarray, list], Union[np.ndarray, list],
    Union[np.ndarray, list], Union[np.ndarray, list], Union[np.ndarray, list],
    np.ndarray, np.ndarray
]:
    if memory_efficient:
        # 内存节省模式：使用生成器和内存映射
        left_frames = list(
            load_video_frames(person_info.left_video_path, as_generator=True)
        )
        right_frames = list(
            load_video_frames(person_info.right_video_path, as_generator=True)
        )
        front_frames = list(
            load_video_frames(person_info.front_video_path, as_generator=True)
        )
        left_2d_kpt = list(
            load_2d_keypoints(person_info.left_2d_kpt_path, as_generator=True)
        )
        right_2d_kpt = list(
            load_2d_keypoints(person_info.right_2d_kpt_path, as_generator=True)
        )
        front_2d_kpt = list(
            load_2d_keypoints(person_info.front_2d_kpt_path, as_generator=True)
        )
        fused_3d_kpt = load_fused_3d_kpt(
            person_info.fused_3d_kpt_path, use_memmap=True
        )
        fused_smoothed_3d_kpt = load_fused_3d_kpt(
            person_info.fused_smoothed_3d_kpt_path, use_memmap=True
        )
    else:
        # 原有模式：一次性加载所有数据
        left_frames = load_video_frames(person_info.left_video_path)
        right_frames = load_video_frames(person_info.right_video_path)
        front_frames = load_video_frames(person_info.front_video_path)
        left_2d_kpt = load_2d_keypoints(person_info.left_2d_kpt_path)
        right_2d_kpt = load_2d_keypoints(person_info.right_2d_kpt_path)
        front_2d_kpt = load_2d_keypoints(person_info.front_2d_kpt_path)
        fused_3d_kpt = load_fused_3d_kpt(person_info.fused_3d_kpt_path)
        fused_smoothed_3d_kpt = load_fused_3d_kpt(
            person_info.fused_smoothed_3d_kpt_path
        )

    return (
        left_frames,
        right_frames,
        front_frames,
        left_2d_kpt,
        right_2d_kpt,
        front_2d_kpt,
        fused_3d_kpt,
        fused_smoothed_3d_kpt,
    )

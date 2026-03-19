#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/frame_visualization.py
Project: /workspace/code/triangulation/vis
Created Date: Tuesday October 14th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 14th 2025 10:57:18 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging

logger = logging.getLogger(__name__)

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
TES = mp_face_mesh.FACEMESH_TESSELATION
CON = mp_face_mesh.FACEMESH_CONTOURS


def draw_and_save_mesh_from_frame(
    frame: np.ndarray,
    mesh: np.ndarray,
    save_path: Path,
    color=(0, 255, 0),
    radius=2,
    thickness=-1,
    draw_tesselation=True,
    draw_contours=True,
    with_index=False,
):
    """在frame上绘制mesh关键点和连接线，并保存图像。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 转 numpy
    img = frame.copy()
    H, W = img.shape[:2]

    # 将 mesh 转为像素坐标（如果是 0~1 范围）
    if np.max(mesh[:, :2]) <= 1.0:
        mesh_px = mesh.copy()
        mesh_px[:, 0] *= W
        mesh_px[:, 1] *= H
    else:
        mesh_px = mesh

    # 绘制点
    for i, (x, y) in enumerate(mesh_px[:, :2]):
        if np.isnan(x) or np.isnan(y):
            continue
        cv2.circle(img, (int(x), int(y)), radius, color, thickness)
        if with_index:
            cv2.putText(
                img,
                str(i),
                (int(x) + 3, int(y) - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

    # 绘制连接线
    if draw_tesselation or draw_contours:
        for edges, col in [(TES, (0, 0, 0)), (CON, (0, 0, 255))]:
            if (edges is TES and not draw_tesselation) or (
                edges is CON and not draw_contours
            ):
                continue
            for a, b in edges:
                if a >= len(mesh_px) or b >= len(mesh_px):
                    continue
                pa, pb = mesh_px[a], mesh_px[b]
                if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
                    continue
                cv2.line(
                    img,
                    tuple(pa[:2].astype(int)),
                    tuple(pb[:2].astype(int)),
                    col,
                    1,
                    cv2.LINE_AA,
                )

    cv2.imwrite(str(save_path), img)
    logger.info(f"Saved mesh visualization frame to {save_path}")
    return img

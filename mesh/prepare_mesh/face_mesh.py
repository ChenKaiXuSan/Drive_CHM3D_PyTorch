#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/prepare_mesh/face_mesh.py
Project: /workspace/code/prepare_mesh
Created Date: Thursday July 17th 2025
Author: Kaixu Chen
-----
Comment:
This module uses MediaPipe FaceMesh to process video frames, extracting 3D face landmarks and rendering face meshes.
It supports video processing, saving results in structured directories.

Have a good code time :)
-----
Last Modified: Thursday July 17th 2025 4:58:47 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-01-2026	Kaixu Chen 去掉处理image的函数，只保留视频处理相关的
"""

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

# ---------- 模块初始化 ----------

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
# MediaPipe 的连接表（tesselation + 轮廓）
TES = mp_face_mesh.FACEMESH_TESSELATION
CON = mp_face_mesh.FACEMESH_CONTOURS


# --- helpers: 坐标&mesh渲染 ---
def landmarks_to_numpy(face_landmarks_list):
    """-> np.array (num_faces, 468, 3) in pixel coords"""
    # * 不normalize，因为之后的图像处理需要像素坐标
    all_faces = []
    for lm in face_landmarks_list:
        pts = [(p.x, p.y, p.z) for p in lm.landmark]
        all_faces.append(pts)
    return np.array(all_faces, dtype=np.float32)


def save_landmarks_to_npz(
    save_info: dict[str, dict[str, np.ndarray]],
    out_path: Path,
):
    """
    将逐帧提取的人脸关键点保存到 .npz 文件

    参数:
        all_landmarks: list，每一帧的 landmark (形状 (F,468,3))，可能为 None
        out_path: 保存路径 (建议用 Path("coords.npz"))
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_path, **save_info)


def render_mesh_rgba(image_shape, face_landmarks_list, line_thick=1, transparent=True):
    """返回 RGBA 图像，只画 mesh（背景透明或黑）"""
    h, w = image_shape[:2]

    # 临时 BGR 画布
    tmp = np.zeros((h, w, 3), dtype=np.uint8)
    if not face_landmarks_list:
        return np.dstack([tmp, np.zeros((h, w), dtype=np.uint8)])

    # 在一个临时 BGR 画布上画线，再转到 RGBA 作为前景
    for lm in face_landmarks_list:
        pts = [(int(p.x * w), int(p.y * h)) for p in lm.landmark]
        # 画 tesselation
        for a, b in TES:
            pa, pb = pts[a], pts[b]
            cv2.line(tmp, pa, pb, (0, 0, 0), line_thick, cv2.LINE_AA)  # 黑色线
        # 画轮廓
        for a, b in CON:
            pa, pb = pts[a], pts[b]
            cv2.line(tmp, pa, pb, (255, 0, 0), line_thick + 1, cv2.LINE_AA)  # 红色线

    # 转 RGBA：把非黑像素作为前景，设 alpha
    rgba = np.dstack([tmp, np.zeros((h, w), dtype=np.uint8)])
    mask = (tmp[:, :, 0] > 0) | (tmp[:, :, 1] > 0) | (tmp[:, :, 2] > 0)
    rgba[:, :, 3] = np.where(mask, 255, 0 if transparent else 255)
    return rgba


def save_mesh_png(rgba_img, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # OpenCV 写 PNG 要 BGRA
    bgra = cv2.cvtColor(rgba_img, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(str(out_path), bgra)


# ---------- 图像处理相关 ----------


def predict_face_mesh(image_rgb, model) -> Optional[list]:
    """执行 FaceMesh 预测"""
    results = model.process(image_rgb)
    return results.multi_face_landmarks


def draw_face_mesh(image_bgr, face_landmarks) -> any:
    """在图像上绘制 mesh（contours + tessellation）"""
    for landmarks in face_landmarks:
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        mp_drawing.draw_landmarks(
            image=image_bgr,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
    return image_bgr


# ---------- 视频处理 ----------


def process_video_with_face_mesh(
    video_path: Path,
    output_path: Optional[Path] = None,
    save_npz: bool = True,
    max_faces: int = 1,
):
    """逐帧处理视频中的人脸 mesh，可选保存输出视频"""
    # 跳过drive view视频
    if "dive" in video_path.name.lower():
        print(f"[!] Skipping drive view video: {video_path}")
        return

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    # prepare directories and video writer
    writer_video = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # video path
        output_video_path = output_path.parent / "video" / output_path.name
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        writer_video = cv2.VideoWriter(
            str(output_video_path), fourcc, fps, (width, height)
        )

        # npz path
        out_npz_path = output_path.parent / "npz" / (output_path.stem + ".npz")
        out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        refine_landmarks=True,
    ) as model:
        save_info = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = predict_face_mesh(rgb_frame, model)

            # 没有的情况下保存空透明图
            mesh_rgba = render_mesh_rgba(
                frame.shape, landmarks, line_thick=1, transparent=True
            )

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            if landmarks:
                # 仅保存mesh帧（透明 PNG）
                mesh_png_path = output_path.parent / "mesh_img" / (video_path.stem)
                mesh_png_path.mkdir(parents=True, exist_ok=True)
                save_mesh_png(mesh_rgba, mesh_png_path / f"{frame_id:06d}.png")

                frame = draw_face_mesh(frame, landmarks)

            if writer_video:
                writer_video.write(frame)

            # save landmarks to npz
            save_info[str(frame_id)] = {
                "raw_frame": rgb_frame,
                "mesh": (
                    (landmarks_to_numpy(landmarks))
                    if landmarks is not None
                    else np.full((1, 478, 3), np.nan)
                ),
                "video_info": {
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "width": width,
                    "height": height,
                },
                "video_path": str(video_path),
            }

            # if frame_id > 20:  # 测试时只跑20帧
            #     break

    cap.release()

    if writer_video:
        writer_video.release()
        print(f"[✓] Saved video output to {output_path}")

    if save_npz and save_info:
        save_landmarks_to_npz(save_info, out_path=out_npz_path)
        print(f"[✓] Saved landmarks to {out_npz_path}")

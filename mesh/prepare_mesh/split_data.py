#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/prepare_mesh/split_data.py
Project: /workspace/code/prepare_mesh
Created Date: Saturday October 18th 2025
Author: Kaixu Chen
-----
Comment:
Split a multi-view video into four sub-videos (left/right/drive_view/front)
directly, without saving PNG frames, and also convert the original video
to H.264 MP4.

Have a good code time :)
-----
Last Modified: Saturday October 18th 2025 10:32:21 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-01-2026	Kaixu Chen	整理代码，保证帧的顺序一致
"""

from pathlib import Path
from tqdm import tqdm
import cv2
import subprocess


def convert_to_h264_mp4(input_video: Path, output_video: Path, overwrite: bool = True):
    """
    把任意视频转成 Label Studio / 浏览器友好的 MP4 格式：
    - 容器：MP4
    - 视频：mpeg4（内置编码器），yuv420p
    - 音频：aac（不支持就改成 copy）
    """
    output_video.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_video.exists():
        output_video.unlink()

    cmd = [
        "ffmpeg",
        "-y",  # 覆盖输出
        "-i",
        str(input_video),
        "-qscale:v",
        "3",  # 1~5 质量不错，1最好，31最差
        "-pix_fmt",
        "yuv420p",
        # 音频：先尝试 aac，不行的话可以改成 copy
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        # 让视频更适合流式播放（可选，但建议）
        "-movflags",
        "+faststart",
        str(output_video),
    ]

    print("[FFMPEG]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[INFO] Converted to MP4 (Label Studio friendly) → {output_video}")


def split_video_to_quadrants(
    video_path: Path,
    save_root: Path,
    crop_bottom: int = 30,
    crop_left: int = 47,
    crop_right: int = 10,
):
    """
    直接从视频划分成四个视角子视频（left/right/dive_view/front），不经过图片中转。
    """

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return

    # 原始 FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # fallback

    # 读一帧拿到尺寸
    ret, frame = cap.read()
    if not ret:
        print(f"[WARN] Empty video: {video_path}")
        cap.release()
        return

    # 按你原来的裁剪方式先 crop
    frame = frame[:-crop_bottom, crop_left:-crop_right]
    height, width = frame.shape[:2]
    mid_x, mid_y = width // 2, height // 2

    # 每个子视角的尺寸
    h_top, h_bottom = mid_y, height - mid_y
    w_left, w_right = mid_x, width - mid_x

    # 这里假定四个区域尺寸一样/近似，如果不是，也可以分别算
    # 为简单起见：按上左块的尺寸建 writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 容器是 mp4，编码实际依赖后端 ffmpeg
    # 若你本地 OpenCV 的 ffmpeg 支持 H.264，可以尝试：cv2.VideoWriter_fourcc(*"avc1") 或 ("H","2","6","4")

    writers = {}

    def make_writer(name, w, h):
        out_dir = save_root
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}.mp4"
        return out_path, cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # 四个视角的视频 writer
    out_left_path, writer_left = make_writer("left", w_left, h_top)
    out_right_path, writer_right = make_writer("right", w_right, h_top)
    out_drive_path, writer_drive = make_writer("drive_view", w_left, h_bottom)
    out_front_path, writer_front = make_writer("front", w_right, h_bottom)

    writers["left"] = writer_left
    writers["right"] = writer_right
    writers["drive_view"] = writer_drive
    writers["front"] = writer_front

    # 第一帧已经读了，先处理
    frame_idx = 0
    while True:
        # 已经裁剪好的 frame
        img = frame

        # 划分四个区域
        img_left = img[0:mid_y, 0:mid_x]
        img_right = img[0:mid_y, mid_x:width]
        img_drive = img[mid_y:height, 0:mid_x]
        img_front = img[mid_y:height, mid_x:width]

        writer_left.write(img_left)
        writer_right.write(img_right)
        writer_drive.write(img_drive)
        writer_front.write(img_front)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"[INFO] {video_path.name}: wrote {frame_idx} frames")

        # 再读下一帧
        ret, frame = cap.read()
        if not ret:
            break
        frame = frame[:-crop_bottom, crop_left:-crop_right]

    cap.release()
    for name, w in writers.items():
        w.release()

    print(
        f"[INFO] Split done for {video_path.name} → "
        f"{out_left_path}, {out_right_path}, {out_drive_path}, {out_front_path}"
    )


def process_videos(
    data_root: Path,
    split_video_output_path: Path,
    original_video_output_path: Path,
):
    """
    data_root:
        person_01/
            scene_a.mpg
            scene_b.mpg
        person_02/
            ...
    split_video_output_path:
        person_01/scene_a/left/scene_a_left.mp4
                            /right/...
                            /dive_view/...
                            /front/...
    original_video_output_path:
        person_01/scene_a_h264.mp4
    """
    for person_dir in tqdm(
        data_root.iterdir(), desc="Processing person directories", ncols=100
    ):
        if not person_dir.is_dir():
            continue

        for video_file in tqdm(
            person_dir.iterdir(),
            desc=f"Processing videos in {person_dir.name}",
            ncols=100,
        ):
            if video_file.stem.startswith("._"):
                continue
            if not (
                video_file.is_file()
                and video_file.suffix.lower() in [".mpg", ".avi", ".mov", ".mp4"]
            ):
                continue

            print(f"[INFO] Processing: {video_file}")

            # 1) 先做原始视频转 H.264 MP4
            out_orig_mp4 = (
                original_video_output_path
                / person_dir.stem
                / f"{video_file.stem}_h264.mp4"
            )
            out_orig_mp4.parent.mkdir(parents=True, exist_ok=True)
            convert_to_h264_mp4(video_file, out_orig_mp4)

            # 2) 再从原始（或 H.264 后）视频划分成四个子视角视频
            split_root = split_video_output_path / person_dir.stem / video_file.stem
            split_video_to_quadrants(video_file, split_root)

            print(f"[INFO] Finished processing: {video_file}")


if __name__ == "__main__":
    # 原始数据根目录
    root_path = Path("/workspace/data/raw")

    # 划分后子视角视频的保存路径
    split_video_output_path = Path("/workspace/data/videos_split")

    # 原始视频转 H.264 MP4 后的保存路径
    # * 这个是为了方便在 Label Studio 或浏览器里查看视频，不用于后续处理
    original_video_output_path = Path("/workspace/data/videos_h264")

    split_video_output_path.mkdir(parents=True, exist_ok=True)
    original_video_output_path.mkdir(parents=True, exist_ok=True)

    process_videos(root_path, split_video_output_path, original_video_output_path)

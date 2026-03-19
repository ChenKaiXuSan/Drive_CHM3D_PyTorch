#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/main.py
Project: /workspace/code
Created Date: Thursday July 17th 2025
Author: Kaixu Chen
-----
Comment:
Use prepare_mesh/face_mesh.py to process video frames for 3D face mesh reconstruction.
And save the results in a structured directory.

The structure is as follows:
save_info[str(frame_id)] = {
                "raw_frame": rgb_frame,
                "mesh": (
                    landmarks_to_numpy(landmarks, frame.shape) if landmarks else None
                ),
            }

Have a good code time :)
-----
Last Modified: Thursday July 17th 2025 5:03:00 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------

26-01-2026	Kaixu Chen	删除处理image的逻辑，只保留video处理
"""

from pathlib import Path

from tqdm import tqdm

from prepare_mesh.face_mesh import process_video_with_face_mesh


def process(path, result_path=None):
    if path.suffix.lower() in [".mp4"]:
        process_video_with_face_mesh(path, save_npz=True, output_path=result_path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


if __name__ == "__main__":
    video_input_dir = Path("/workspace/data/videos_split")
    result_dir = Path("/workspace/data/mesh")

    for one_person in tqdm(
        video_input_dir.iterdir(), desc="Processing person", ncols=100
    ):
        if not one_person.is_dir():
            continue

        for video_dir in tqdm(
            one_person.iterdir(), desc="Processing one person videos", ncols=100
        ):
            if not video_dir.is_dir():
                continue

            for video_file in video_dir.glob("*.mp4"):
                _res_path = result_dir / one_person.name / video_dir.name

                print(f"Processing video: {video_file}")
                process(path=video_file, result_path=_res_path / video_file.name)
                print(f"Finished processing video: {video_file}")

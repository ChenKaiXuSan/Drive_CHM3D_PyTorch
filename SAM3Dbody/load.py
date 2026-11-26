#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/SAM3Dbody/load.py
Project: /workspace/code/SAM3Dbody
Created Date: Friday January 23rd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 23rd 2026 4:50:57 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from common.frame_rate import annotation_to_camera_frame

logger = logging.getLogger(__name__)


def load_data(input_video_path: Dict[str, Path]) -> Dict[str, List[np.ndarray]]:
    """
    動画ファイルからすべてのフレームを読み込み、RGB形式のリストとして返します。

    引数:
        input_video_path: 動画ファイルのパスの辞書 (キーは視点名、値は Path オブジェクト)

    戻り値:
        List[np.ndarray]: 全フレームのリスト。各フレームは RGB 形式の numpy 配列。
    """
    view_frames_list: Dict[str, List[np.ndarray]] = {}

    for view, video_path in input_video_path.items():

        # 2. 動画のキャプチャ開始
        logger.info(f"動画ファイルからフレームを抽出中: {video_path}")
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"エラー: 動画を開くことができませんでした -> {video_path}")
            return []

        # 3. 全フレームをループで読み込み
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # OpenCVはBGR形式なのでRGBに変換
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                view_frames_list.setdefault(view, []).append(frame_rgb)
        finally:
            cap.release()

        logger.info(
            f"動画の読み込み完了。合計 {len(view_frames_list[view])} フレームを抽出しました。"
        )
    return view_frames_list


MAPPING = {
    "night_high": "夜多い",
    "night_low": "夜少ない",
    "day_high": "昼多い",
    "day_low": "昼少ない",
}


def get_annotation_dict(file_path):
    """根据start mid end指定推理范围

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 格納用の辞書
    # 構造: { "person_01": { "昼多い": {"start": 0, "mid": 0, "end": 0}, ... }, ... }
    master_dict = {}

    for item in data:
        # ファイル名から情報の抽出
        video_path = item.get("video", "")
        file_name = os.path.basename(video_path)
        parts = file_name.split("_")

        if len(parts) < 4:
            continue

        person = f"{parts[0]}_{parts[1]}"  # person_01
        env = MAPPING.get(
            f"{parts[2]}_{parts[3]}", f"{parts[2]}_{parts[3]}"
        )  # day_high

        # フレーム情報の取得
        frames = {"start": None, "mid": None, "end": None}
        video_labels = item.get("videoLabels", [])

        for label_obj in video_labels:
            labels = label_obj.get("timelinelabels", [])
            if labels:
                label_name = labels[0]
                frame_num = label_obj.get("ranges", [{}])[0].get("start")
                if label_name in frames and frame_num is not None:
                    # 标注是 24 fps 帧号，推理按相机帧号
                    frames[label_name] = annotation_to_camera_frame(frame_num)

        # 辞書への格納
        if person not in master_dict:
            master_dict[person] = {}

        master_dict[person][env] = frames

    return master_dict

#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/bundle_adjustment/visualization/skeleton_visualizer copy.py
Project: /workspace/code/bundle_adjustment/visualization
Created Date: Sunday December 7th 2025
Author: Kaixu Chen
-----
Comment:
以人物的地面为世界中心
左相机默认的pred cam t
右相机是按照刚体对其准的

相机看向人物中心，根据focal_length计算FOV

Have a good code time :)
-----
Last Modified: Sunday December 7th 2025 2:20:39 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .utils import parse_pose_metainfo


class SceneVisualizer:
    def __init__(
        self,
        bbox_color: Optional[Union[str, Tuple[int]]] = "green",
        kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = "red",
        link_color: Optional[Union[str, Tuple[Tuple[int]]]] = None,
        text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
        line_width: Union[int, float] = 1,
        radius: Union[int, float] = 3,
        alpha: float = 1.0,
        show_keypoint_weight: bool = False,
    ):
        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight

        # Pose specific meta info if available.
        self.pose_meta = {}
        self.skeleton = None

    def set_pose_meta(self, pose_meta: Dict):
        parsed_meta = parse_pose_metainfo(pose_meta)

        self.pose_meta = parsed_meta.copy()
        self.bbox_color = parsed_meta.get("bbox_color", self.bbox_color)
        self.kpt_color = parsed_meta.get("keypoint_colors", self.kpt_color)
        self.link_color = parsed_meta.get("skeleton_link_colors", self.link_color)
        self.skeleton = parsed_meta.get("skeleton_links", self.skeleton)

    # ---------- 主函数：画人物 + 左右相机 + 视锥体 ----------
    def draw_scene(
        self,
        ax: plt.axes = None,
        kpts_world: np.ndarray = np.array([]),
        elev=-30,
        azim=270,
    ):
        """
        在给定的 ax 上画 3D 场景；如果 ax 为 None，则自己新建 fig+ax。
        """
        kpts_world = np.asarray(kpts_world)

        created_fig = None
        if ax is None:
            created_fig = plt.figure(figsize=(8, 8))
            ax = created_fig.add_subplot(111, projection="3d")

        # --- 颜色处理 ---
        kpt_color = "r"
        raw_kpt_colors_mp = self.kpt_color
        if raw_kpt_colors_mp is not None and not isinstance(raw_kpt_colors_mp, str):
            kpt_color = np.array(raw_kpt_colors_mp, dtype=np.float32) / 255.0

        link_color = "b"
        raw_link_colors_mp = self.link_color
        if raw_link_colors_mp is not None and not isinstance(raw_link_colors_mp, str):
            link_color = np.array(raw_link_colors_mp, dtype=np.float32) / 255.0

        # --- 1. 人体骨架 ---
        ax.scatter(
            kpts_world[:, 0],
            kpts_world[:, 1],
            kpts_world[:, 2],
            c=kpt_color,
            marker="o",
            s=self.radius * 10,
            alpha=self.alpha,
        )

        if self.skeleton is not None:
            link_colors_mp = link_color
            for i, (p1_idx, p2_idx) in enumerate(self.skeleton):
                p1 = kpts_world[p1_idx]
                p2 = kpts_world[p2_idx]
                color = (
                    link_colors_mp[i % len(link_colors_mp)]
                    if not isinstance(link_colors_mp, str)
                    else link_colors_mp
                )
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color,
                    linewidth=self.line_width * 2,
                    alpha=self.alpha,
                )

        # ax.set_xlim3d(-0.1, 0.1)
        # ax.set_ylim3d(-1.5, 0)
        # ax.set_zlim3d(-0.1, 0.1)

        # 标记世界坐标系原点
        ax.scatter([0], [0], [0], s=60)
        ax.text(0, 0, 0, "world center (0,0,0)")

        ax.set_xlim3d(-0.3, 0.3)
        ax.set_zlim3d(0, 1.8)
        ax.set_ylim3d(-0.3, 0.3)

        ax.set_box_aspect((1, 1, 1))

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.view_init(elev=elev, azim=azim)

        return created_fig if created_fig is not None else ax

    def draw_frame_with_scene(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        pose_3d: np.ndarray,  # (J,3)
        frame_num: int = 0,
    ):
        """
        渲染一个 frame：左图+右图+3D pose，并返回 figure。
        """

        fig = plt.figure(figsize=(10, 10))
        fig.suptitle(f"Frame {frame_num}")
        gs = GridSpec(2, 3, figure=fig)

        # -------- 左视角 ---------- #
        axL = fig.add_subplot(gs[0, 0])
        axL.imshow(left_frame)
        axL.axis("off")
        axL.set_title("Left view")

        # -------- 右视角 ---------- #
        axR = fig.add_subplot(gs[1, 0])
        axR.imshow(right_frame)
        axR.axis("off")
        axR.set_title("Right view")

        # -------- 3D pose ---------- #
        # 交换y和z轴，并将z轴取反，使得y轴朝上，z轴朝前（相机看向人物）
        # pose_3d[:, [1, 2]] = pose_3d[:, [2, 1]]
        # pose_3d[:, 2] = -pose_3d[:, 2]

        ax_3d_left = fig.add_subplot(gs[0, 1], projection="3d")
        ax_3d_left.set_title("left side view")
        self.draw_scene(
            kpts_world=pose_3d,
            ax=ax_3d_left,
            elev=0,
            azim=0,
        )

        ax_3d_right = fig.add_subplot(gs[1, 1], projection="3d")
        ax_3d_right.set_title("right side view")
        self.draw_scene(
            kpts_world=pose_3d,
            ax=ax_3d_right,
            elev=0,
            azim=-180,
        )

        ax_3d_top_left = fig.add_subplot(gs[0, 2], projection="3d")
        ax_3d_top_left.set_title("top left view")
        self.draw_scene(
            kpts_world=pose_3d,
            ax=ax_3d_top_left,
            elev=90,
            azim=0,
        )

        ax_3d_top_right = fig.add_subplot(gs[1, 2], projection="3d")
        ax_3d_top_right.set_title("top right view")
        self.draw_scene(
            kpts_world=pose_3d,
            ax=ax_3d_top_right,
            elev=90,
            azim=-180,
        )

        fig.tight_layout()
        return fig

    def save(
        self,
        image: plt.figure,
        save_path: Path,
    ):
        """Save the drawn image to disk.

        Args:
            image (np.ndarray): The drawn image.
            save_path (str): The path to save the image.
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # image.savefig(save_path, dpi=300)
        image.savefig(
            save_path,
            dpi=300,
            facecolor="white",  # 背景白
            edgecolor="white",
        )

        plt.close(image)

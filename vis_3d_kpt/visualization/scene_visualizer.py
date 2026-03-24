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
from typing import Dict, List, Optional, Tuple, Union

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

    def get_skeleton_edges_with_colors(
        self,
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float, float]]]:
        """Extract skeleton edges and RGB colors in [0, 1] from pose metadata."""

        edges: List[Tuple[int, int]] = []
        colors: List[Tuple[float, float, float]] = []

        skeleton_info = self.pose_meta.get("skeleton_info")
        keypoint_info = self.pose_meta.get("keypoint_info")

        if skeleton_info is not None and keypoint_info is not None:
            name_to_idx = {
                item["name"]: idx for idx, item in keypoint_info.items() if "name" in item
            }
            for _, bone_info in skeleton_info.items():
                joint1_name, joint2_name = bone_info["link"]
                if joint1_name in name_to_idx and joint2_name in name_to_idx:
                    edges.append((name_to_idx[joint1_name], name_to_idx[joint2_name]))
                    colors.append(tuple(c / 255.0 for c in bone_info["color"]))
            return edges, colors

        if self.skeleton is not None:
            edges = [tuple(map(int, link)) for link in self.skeleton]

        raw_link_colors = self.link_color
        if raw_link_colors is None:
            colors = [(0.0, 0.0, 1.0)] * len(edges)
        elif isinstance(raw_link_colors, str):
            colors = [raw_link_colors] * len(edges)
        else:
            normalized = np.asarray(raw_link_colors, dtype=np.float32) / 255.0
            colors = [tuple(color.tolist()) for color in normalized]

        return edges, colors

    def draw_skeleton_edges(
        self,
        ax: plt.axes,
        keypoints_3d: np.ndarray,
        edges: Optional[List[Tuple[int, int]]] = None,
        colors: Optional[List[Tuple[float, float, float]]] = None,
        part_index_set: Optional[set] = None,
        linewidth: float = 2.6,
        alpha: float = 0.85,
    ):
        """Draw skeleton edges on a 3D axis."""

        if edges is None or colors is None:
            edges, colors = self.get_skeleton_edges_with_colors()

        color_count = len(colors)
        for i, (idx1, idx2) in enumerate(edges):
            if idx1 >= len(keypoints_3d) or idx2 >= len(keypoints_3d):
                continue
            if part_index_set is not None and (
                idx1 not in part_index_set or idx2 not in part_index_set
            ):
                continue

            p1 = keypoints_3d[idx1]
            p2 = keypoints_3d[idx2]
            color = colors[i % color_count] if color_count > 0 else "b"
            ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
            )

    def visualize_3d_skeleton(
        self,
        keypoints_3d: np.ndarray,
        part_indices: Optional[List[int]] = None,
        title: str = "3D Skeleton Visualization",
        figsize: Tuple[int, int] = (12, 10),
    ):
        """Visualize a single-frame 3D skeleton."""

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if part_indices is not None:
            keypoints = keypoints_3d[part_indices]
        else:
            keypoints = keypoints_3d
            part_indices = list(range(len(keypoints_3d)))

        self.draw_skeleton_edges(
            ax=ax,
            keypoints_3d=keypoints_3d,
            part_index_set=set(part_indices),
            linewidth=2.6,
            alpha=0.85,
        )

        ax.scatter(
            keypoints[:, 0],
            keypoints[:, 1],
            keypoints[:, 2],
            c="red",
            marker="o",
            s=50,
            label="Keypoints",
        )

        for i, (x, y, z) in enumerate(keypoints):
            ax.text(x, y, z, f"{part_indices[i]}", fontsize=8)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(title)
        ax.legend()
        ax.view_init(elev=20, azim=45)

        return fig, ax

    # ---------- 主函数：画人物 + 左右相机 + 视锥体 ----------
    def draw_scene(
        self,
        ax: plt.axes = None,
        kpts_world: np.ndarray = np.array([]),
        elev=-30,
        azim=270,
        part_indices: Optional[List[int]] = None,
    ):
        """
        在给定的 ax 上画 3D 场景；如果 ax 为 None，则自己新建 fig+ax。
        """
        kpts_world = np.asarray(kpts_world)

        created_fig = None
        if ax is None:
            created_fig = plt.figure(figsize=(8, 8))
            ax = created_fig.add_subplot(111, projection="3d")

        kpt_color = "r"
        raw_kpt_colors_mp = self.kpt_color
        if raw_kpt_colors_mp is not None and not isinstance(raw_kpt_colors_mp, str):
            kpt_color = np.array(raw_kpt_colors_mp, dtype=np.float32) / 255.0

        if part_indices is not None:
            keypoints = kpts_world[part_indices]
            part_index_set = set(part_indices)
            if not isinstance(kpt_color, str) and len(kpt_color) >= len(kpts_world):
                kpt_color = kpt_color[part_indices]
        else:
            keypoints = kpts_world
            part_index_set = None

        self.draw_skeleton_edges(
            ax=ax,
            keypoints_3d=kpts_world,
            part_index_set=part_index_set,
            linewidth=self.line_width * 2,
            alpha=self.alpha,
        )

        ax.scatter(
            keypoints[:, 0],
            keypoints[:, 1],
            keypoints[:, 2],
            c=kpt_color,
            marker="o",
            s=self.radius * 10,
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
        front_frame: np.ndarray,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        pose_3d: np.ndarray,  # (J,3)
        frame_num: int = 0,
        part_indices: Optional[List[int]] = None,
    ):
        """
        渲染一个 frame：左图+右图+3D pose，并返回 figure。
        """

        fig = plt.figure(figsize=(14, 9))
        fig.suptitle(f"Frame {frame_num}")
        gs = GridSpec(2, 4, figure=fig)

        # -------- 前视角 ---------- #
        axF = fig.add_subplot(gs[0, 0])
        axF.imshow(front_frame)
        axF.axis("off")
        axF.set_title("Front view")

        # -------- 左视角 ---------- #
        axL = fig.add_subplot(gs[0, 1])
        axL.imshow(left_frame)
        axL.axis("off")
        axL.set_title("Left view")

        # -------- 右视角 ---------- #
        axR = fig.add_subplot(gs[0, 2])
        axR.imshow(right_frame)
        axR.axis("off")
        axR.set_title("Right view")

        # 占位，保持上排视觉平衡
        ax_pad = fig.add_subplot(gs[0, 3])
        ax_pad.axis("off")

        # -------- 3D pose ---------- #
        # 交换y和z轴，并将z轴取反，使得y轴朝上，z轴朝前（相机看向人物）
        # pose_3d[:, [1, 2]] = pose_3d[:, [2, 1]]
        # pose_3d[:, 2] = -pose_3d[:, 2]

        ax_3d_left = fig.add_subplot(gs[1, 0], projection="3d")
        ax_3d_left.set_title("left side view")
        self.draw_scene(
            kpts_world=pose_3d,
            ax=ax_3d_left,
            elev=0,
            azim=0,
            part_indices=part_indices,
        )

        ax_3d_right = fig.add_subplot(gs[1, 1], projection="3d")
        ax_3d_right.set_title("right side view")
        self.draw_scene(
            kpts_world=pose_3d,
            ax=ax_3d_right,
            elev=0,
            azim=-180,
            part_indices=part_indices,
        )

        ax_3d_top_left = fig.add_subplot(gs[1, 2], projection="3d")
        ax_3d_top_left.set_title("top left view")
        self.draw_scene(
            kpts_world=pose_3d,
            ax=ax_3d_top_left,
            elev=90,
            azim=0,
            part_indices=part_indices,
        )

        ax_3d_top_right = fig.add_subplot(gs[1, 3], projection="3d")
        ax_3d_top_right.set_title("top right view")
        self.draw_scene(
            kpts_world=pose_3d,
            ax=ax_3d_top_right,
            elev=90,
            azim=-180,
            part_indices=part_indices,
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

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .utils import draw_text, parse_pose_metainfo


class SkeletonVisualizer:
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

    def draw_skeleton(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        kpt_thr: float = 0.3,
        show_kpt_idx: bool = False,
        part_indices: Optional[list] = None,
    ):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            keypoints (np.ndarray): B x N x 3
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``
            part_indices (Optional[list]): If provided, only draw keypoints
                at these indices (e.g., [0, 1, 2, 3, 4, 5, 6, 69] for head).
                Default: ``None`` (draw all keypoints)

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        image = image.copy()
        img_h, img_w, _ = image.shape
        if len(keypoints.shape) == 2:
            keypoints = keypoints[None, :, :]

        # ensure keypoints has shape B x N x 3
        if keypoints.shape[2] != 3:
            keypoints = np.concatenate(
                [keypoints, np.ones((*keypoints.shape[:2], 1))], axis=2
            )

        # loop for each person
        for cur_keypoints in keypoints:
            kpts = cur_keypoints[:, :-1]
            score = cur_keypoints[:, -1]

            if self.kpt_color is None or isinstance(self.kpt_color, str):
                kpt_color = [self.kpt_color] * len(kpts)
            elif len(self.kpt_color) <= len(kpts):
                kpt_color = self.kpt_color
            else:
                raise ValueError(
                    f"the length of kpt_color "
                    f"({len(self.kpt_color)}) is larger than "
                    f"that of keypoints ({len(kpts)})"
                )

            # draw links
            if self.skeleton is not None and self.link_color is not None:
                if self.link_color is None or isinstance(self.link_color, str):
                    link_color = [self.link_color] * len(self.skeleton)
                elif len(self.link_color) <= len(self.skeleton):
                    link_color = self.link_color
                else:
                    raise ValueError(
                        f"the length of link_color "
                        f"({len(self.link_color)}) is larger than "
                        f"that of skeleton links ({len(self.skeleton)})"
                    )

                for sk_id, sk in enumerate(self.skeleton):
                    # Skip links if part_indices is specified and endpoints not in part
                    if part_indices is not None:
                        if sk[0] not in part_indices or sk[1] not in part_indices:
                            continue

                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    if (
                        pos1[0] <= 0
                        or pos1[0] >= img_w
                        or pos1[1] <= 0
                        or pos1[1] >= img_h
                        or pos2[0] <= 0
                        or pos2[0] >= img_w
                        or pos2[1] <= 0
                        or pos2[1] >= img_h
                        or score[sk[0]] < kpt_thr
                        or score[sk[1]] < kpt_thr
                        or link_color[sk_id] is None
                    ):
                        # skip the link that should not be drawn
                        continue

                    color = link_color[sk_id]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(
                            0, min(1, 0.5 * (score[sk[0]] + score[sk[1]]))
                        )

                    image = cv2.line(
                        image,
                        pos1,
                        pos2,
                        color,
                        thickness=self.line_width,
                    )

            # draw each point on image
            for kid, kpt in enumerate(kpts):
                # Skip keypoint if part_indices is specified and kid not in part
                if part_indices is not None and kid not in part_indices:
                    continue

                if score[kid] < kpt_thr or kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = kpt_color[kid]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)
                transparency = self.alpha
                if self.show_keypoint_weight:
                    transparency *= max(0, min(1, score[kid]))

                if transparency == 1.0:
                    image = cv2.circle(
                        image,
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                else:
                    temp = image = cv2.circle(
                        image.copy(),
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        color,
                        -1,
                    )
                    image = cv2.addWeighted(
                        image, 1 - transparency, temp, transparency, 0
                    )

                if show_kpt_idx:
                    kpt[0] += self.radius
                    kpt[1] -= self.radius
                    image = draw_text(
                        image,
                        str(kid),
                        kpt,
                        image_size=(img_w, img_h),
                        color=color,
                        font_size=self.radius * 3,
                        vertical_alignment="bottom",
                        horizontal_alignment="center",
                    )

        return image

    def draw_skeleton_3d(
        self,
        ax: plt.axes,
        points_3d: np.ndarray,
        window_title: str = "3D Skeleton Visualization",
        part_indices: Optional[list] = None,
    ):
        """
        使用 matplotlib 绘制 3D 关键点和骨架连线。

        Args:
            ax (plt.axes): Matplotlib 3D 轴对象（可选）。
            points_3d (np.ndarray): N x 3 形状的 NumPy 数组，代表 3D 坐标 (x, y, z)。
            window_title (str): 可视化窗口的标题。
            part_indices (Optional[list]): If provided, only draw keypoints
                at these indices. Default: ``None`` (draw all keypoints)
        """
        # 1. 初始化 3D 绘图
        created_fig = None
        if ax is None:
            created_fig = plt.figure(figsize=(10, 10))
            ax = created_fig.add_subplot(111, projection="3d")

        ax.set_title(window_title)

        # 设置坐标轴标签
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Filter points if part_indices is specified
        if part_indices is not None:
            points_3d_filtered = points_3d[part_indices]
            indices_to_draw = part_indices
        else:
            points_3d_filtered = points_3d
            indices_to_draw = list(range(len(points_3d)))

        # 确保坐标轴比例一致，避免扭曲
        max_range = (
            np.array(
                [
                    points_3d_filtered[:, i].max() - points_3d_filtered[:, i].min()
                    for i in range(3)
                ]
            ).max()
            / 2.0
        )
        mid_x = (
            points_3d_filtered[:, 0].max() + points_3d_filtered[:, 0].min()
        ) / 2.0
        mid_y = (
            points_3d_filtered[:, 1].max() + points_3d_filtered[:, 1].min()
        ) / 2.0
        mid_z = (
            points_3d_filtered[:, 2].max() + points_3d_filtered[:, 2].min()
        ) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 2. 绘制 3D 关键点
        raw_kpt_colors_mp = self.kpt_color
        # 关键点颜色转换
        if raw_kpt_colors_mp is not None and not isinstance(raw_kpt_colors_mp, str):
            # 将颜色转换为 NumPy 数组并归一化到 0.0-1.0 范围
            kpt_color = np.array(raw_kpt_colors_mp, dtype=np.float32) / 255.0
        else:
            kpt_color = (1.0, 0.0, 0.0)  # Default red

        # 连线颜色转换
        raw_link_colors_mp = self.link_color
        if raw_link_colors_mp is not None and not isinstance(raw_link_colors_mp, str):
            link_color = np.array(raw_link_colors_mp, dtype=np.float32) / 255.0
        else:
            link_color = (0.0, 0.0, 1.0)  # Default blue

        ax.scatter(
            points_3d_filtered[:, 0],
            points_3d_filtered[:, 1],
            points_3d_filtered[:, 2],
            c=[kpt_color],
            marker="o",
            s=self.radius * 10,  # 调整点的大小以便在 3D 中可见
            alpha=self.alpha,
        )

        # 3. 绘制 3D 骨架连线
        if self.skeleton is not None:
            for i, (p1_idx, p2_idx) in enumerate(self.skeleton):
                # Skip links if part_indices is specified and endpoints not in part
                if part_indices is not None:
                    if p1_idx not in indices_to_draw or p2_idx not in indices_to_draw:
                        continue

                # 提取两个点的坐标
                p1 = points_3d[p1_idx]
                p2 = points_3d[p2_idx]

                # 绘制连接线
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=link_color,
                    linewidth=self.line_width * 2,  # 调整线宽
                    alpha=self.alpha,
                )

        # ax.view_init(elev=-30, azim=270)

        plt.tight_layout()

        # 返回 Figure 对象，Notebook 会自动显示它
        return created_fig if created_fig is not None else ax

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

        image.savefig(save_path, dpi=300)

        plt.close(image)

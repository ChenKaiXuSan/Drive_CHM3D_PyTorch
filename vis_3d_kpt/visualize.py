#!/usr/bin/env python3

import logging
import gc

from pathlib import Path

from tqdm import tqdm
import numpy as np

from .load import OnePersonInfo, load_helper
from .metadata.mhr70 import pose_info as mhr70_pose_info
from .visualization.save_utils import merge_frame_to_video
from .visualization.scene_visualizer import SceneVisualizer
from .visualization.skeleton_visualizer import SkeletonVisualizer

logger = logging.getLogger(__name__)

DEFAULT_VIEWS = [
    ("front_left", -25, 270),
    ("front_right", -25, 90),
    ("top", 85, 270),
    ("side", 0, 0),
]

SIMPLE_VIEWS = [
    ("perspective", -20, 255),
]


def setup_visualizer():
    """Setup visualizers with default pose meta (MHR70)."""

    skeleton_visualizer = SkeletonVisualizer(line_width=2, radius=5)
    skeleton_visualizer.set_pose_meta(mhr70_pose_info)

    scene_visualizer = SceneVisualizer(line_width=2, radius=5)
    scene_visualizer.set_pose_meta(mhr70_pose_info)

    return skeleton_visualizer, scene_visualizer


def run_visualization(
    person_info: OnePersonInfo,
    out_dir: Path,
) -> None:
    """Run visualization using an argparse.Namespace of options.

    This function contains the core logic originally in `main()` and is
    safe to import and call from other scripts.
    """
    left_frames = None
    right_frames = None
    front_frames = None
    left_2d_kpt = None
    right_2d_kpt = None
    front_2d_kpt = None
    fused_3d_kpt = None
    fused_smoothed_3d_kpt = None

    try:
        out_dir = out_dir.resolve()
        logger.info(
            "Starting visualization for %s to %s", person_info.person_name, str(out_dir)
        )

        (
            left_frames,
            right_frames,
            front_frames,
            left_2d_kpt,
            right_2d_kpt,
            front_2d_kpt,
            fused_3d_kpt,
            fused_smoothed_3d_kpt,
        ) = load_helper(person_info, memory_efficient=True)

        skeleton_visualizer, scene_visualizer = setup_visualizer()

        # * pro的左右视频长度不一致，按最短的来
        frame_count = min(
            len(left_frames),
            len(right_frames),
            len(front_frames),
            len(left_2d_kpt),
            len(right_2d_kpt),
            len(front_2d_kpt),
            fused_3d_kpt.shape[0],
            fused_smoothed_3d_kpt.shape[0],
        )

        for frame_idx in tqdm(range(frame_count), desc="Processing frames"):
            # if frame_idx > 50:  # 先只处理前50帧，测试用
            #     break

            try:
                process_frame(
                    left_frames=left_frames[frame_idx],
                    right_frames=right_frames[frame_idx],
                    front_frames=front_frames[frame_idx],
                    left_2d_kpt=left_2d_kpt[frame_idx],
                    right_2d_kpt=right_2d_kpt[frame_idx],
                    front_2d_kpt=front_2d_kpt[frame_idx],
                    fused_3d_kpt=fused_3d_kpt[frame_idx],
                    fused_smoothed_3d_kpt=fused_smoothed_3d_kpt[frame_idx],
                    frame_idx=frame_idx,
                    out_root=out_dir,
                    skeleton_visualizer=skeleton_visualizer,
                    scene_visualizer=scene_visualizer,
                )
            except Exception as e:
                logger.error("Error processing frame %d: %s", frame_idx, e)
                continue

        logger.info("Visualization completed for %s", person_info.person_name)

        # 合成图片为视频
        merge_frame_to_video(out_dir, "fused", fps=30)
        merge_frame_to_video(out_dir, "smoothed", fps=30)
        merge_frame_to_video(out_dir, "frame_scene", fps=30)

    except Exception as e:
        logger.error("Error in visualization: %s", e)
        raise
    finally:
        # 结束之后清理内存
        try:
            if left_frames is not None:
                del left_frames
            if right_frames is not None:
                del right_frames
            if front_frames is not None:
                del front_frames
            if left_2d_kpt is not None:
                del left_2d_kpt
            if right_2d_kpt is not None:
                del right_2d_kpt
            if front_2d_kpt is not None:
                del front_2d_kpt
            if fused_3d_kpt is not None:
                del fused_3d_kpt
            if fused_smoothed_3d_kpt is not None:
                del fused_smoothed_3d_kpt
            gc.collect()
            logger.info("Memory cleaned up")
        except Exception:
            pass


def process_frame(
    left_frames: np.ndarray,
    right_frames: np.ndarray,
    front_frames: np.ndarray,
    left_2d_kpt: np.ndarray,
    right_2d_kpt: np.ndarray,
    front_2d_kpt: np.ndarray,
    fused_3d_kpt: np.ndarray,
    fused_smoothed_3d_kpt: np.ndarray,
    frame_idx: int,
    out_root: Path,
    skeleton_visualizer: SkeletonVisualizer,
    scene_visualizer: SceneVisualizer,
) -> None:
    # 转变坐标系（SAM的坐标系是右手系，Z轴向前；我们希望的坐标系是Z轴向上）
    fused_3d_kpt = fused_3d_kpt.copy()
    fused_3d_kpt[:, [1, 2]] = fused_3d_kpt[:, [2, 1]]  # swap Y and Z
    fused_3d_kpt[:, 2] *= -1  # invert new Z (old Y) to make it forward-facing

    fused_smoothed_3d_kpt = fused_smoothed_3d_kpt.copy()
    fused_smoothed_3d_kpt[:, [1, 2]] = fused_smoothed_3d_kpt[:, [2, 1]]  # swap Y and Z
    fused_smoothed_3d_kpt[:, 2] *= -1  # invert new Z (old Y) to make it forward-facing

    # ---------- 画骨架图 ----------
    # * 时间优化前的
    _skeleton = skeleton_visualizer.draw_skeleton_3d(ax=None, points_3d=fused_3d_kpt)

    skeleton_visualizer.save(
        image=_skeleton,
        save_path=out_root / "fused" / f"{frame_idx}.png",
    )

    # * 时间优化后的
    _skeleton_smoothed = skeleton_visualizer.draw_skeleton_3d(
        ax=None, points_3d=fused_smoothed_3d_kpt
    )

    skeleton_visualizer.save(
        image=_skeleton_smoothed,
        save_path=out_root / "smoothed" / f"{frame_idx}.png",
    )

    # 画左右frame + 2D骨架

    left_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=left_frames, keypoints=left_2d_kpt
    )
    right_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=right_frames, keypoints=right_2d_kpt
    )
    front_kpt_with_frame = skeleton_visualizer.draw_skeleton(
        image=front_frames, keypoints=front_2d_kpt
    )

    _frame_scene = scene_visualizer.draw_frame_with_scene(
        front_frame=front_kpt_with_frame,
        left_frame=left_kpt_with_frame,
        right_frame=right_kpt_with_frame,
        pose_3d=fused_smoothed_3d_kpt,
    )

    scene_visualizer.save(
        image=_frame_scene,
        save_path=out_root / "frame_scene" / f"{frame_idx}.png",
    )

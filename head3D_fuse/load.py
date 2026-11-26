import dataclasses
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.frame_rate import annotation_to_camera_frame

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PersonInfo:
    person_id: str  # idは '01' などの文字列である可能性が高いためstrに変更
    env_id: str
    front_npz_paths: List[Path] = dataclasses.field(default_factory=list)
    left_npz_paths: List[Path] = dataclasses.field(default_factory=list)
    right_npz_paths: List[Path] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class FrameTriplet:
    frame_idx: int
    npz_paths: Dict[str, Path]


def _extract_frame_idx(npz_path: Path) -> Optional[int]:
    stem = npz_path.stem
    # Expected format: "<frame>_SAM3D_body.npz"; fallback to leading digits and trailing numeric tokens.
    match = re.match(r"^(\d+)_SAM3D", stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.match(r"^(\d+)", stem)
    if match:
        return int(match.group(1))
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    matches = re.findall(r"\d+", stem)
    return int(matches[-1]) if matches else None


def _lookup_annotation_range(
    annotation_dict: Optional[dict], person_id: str, env_id: str
) -> Tuple[Optional[int], Optional[int]]:
    if not annotation_dict:
        return None, None
    for key in (person_id, f"person_{person_id}"):
        if key in annotation_dict and env_id in annotation_dict[key]:
            frames_info = annotation_dict[key][env_id]
            return frames_info.get("start"), frames_info.get("end")
    return None, None


def _collect_view_npz(view_path: Path) -> Dict[int, Path]:
    npz_files = list(view_path.glob("*.npz"))
    if not npz_files:
        logger.warning(f"No npz files found in: {view_path}")
        return {}
    frame_map: Dict[int, Path] = {}
    for npz_file in npz_files:
        frame_idx = _extract_frame_idx(npz_file)
        if frame_idx is None:
            logger.warning(f"Invalid frame number in file: {npz_file}")
            continue
        frame_map[frame_idx] = npz_file
    return frame_map


def assemble_view_npz_paths(
    person_env_dir: Path,
    view_list: List[str],
    annotation_dict: Optional[dict] = None,
    min_views: Optional[int] = None,
) -> Tuple[List[FrameTriplet], Dict[str, Dict[str, List[int]]]]:
    """收集每帧各视角的 npz。

    min_views: 一帧至少要有几个视角才参与融合；None 表示必须所有视角都有。
    视角不全时，FrameTriplet.npz_paths 只包含存在的视角。
    """
    min_views = len(view_list) if min_views is None else int(min_views)
    view_frames: Dict[str, Dict[int, Path]] = {}
    for view in view_list:
        view_frames[view] = _collect_view_npz(person_env_dir / view)

    start_frame, end_frame = _lookup_annotation_range(
        annotation_dict, person_env_dir.parent.name, person_env_dir.name
    )
    for view, frame_map in view_frames.items():
        view_frames[view] = {
            frame_idx: npz_path
            for frame_idx, npz_path in frame_map.items()
            if (start_frame is None or frame_idx >= start_frame)
            and (end_frame is None or frame_idx <= end_frame)
        }

    if view_frames:
        common_frames = set.intersection(
            *[set(frame_map.keys()) for frame_map in view_frames.values()]
        )
        all_frames = set.union(
            *[set(frame_map.keys()) for frame_map in view_frames.values()]
        )
    else:
        common_frames = set()
        all_frames = set()

    frame_triplets = []
    for frame_idx in sorted(all_frames):
        npz_paths = {
            view: view_frames[view][frame_idx]
            for view in view_list
            if frame_idx in view_frames[view]
        }
        if len(npz_paths) >= min_views:
            frame_triplets.append(FrameTriplet(frame_idx=frame_idx, npz_paths=npz_paths))

    report = {
        "frames_per_view": {
            view: sorted(frame_map.keys()) for view, frame_map in view_frames.items()
        },
        "missing_frames": {
            view: sorted(all_frames - set(frame_map.keys()))
            for view, frame_map in view_frames.items()
        },
        "common_frames": sorted(common_frames),
        "start_frame": start_frame,
        "end_frame": end_frame,
    }
    return frame_triplets, report


def load_npz_output(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    if "output" not in data:
        raise KeyError(f"Missing 'output' in {npz_path}")
    output = data["output"]
    if isinstance(output, np.ndarray) and output.dtype == object:
        output = output.item()
    return output


def compare_npz_files(npz_paths: Dict[str, Path]) -> Optional[dict]:
    outputs = {view: load_npz_output(path) for view, path in npz_paths.items()}
    all_keys = set().union(*[set(output.keys()) for output in outputs.values()])
    missing_keys = {
        view: sorted(all_keys - set(output.keys())) for view, output in outputs.items()
    }

    mismatched_shapes: Dict[str, Dict[str, Optional[Tuple[int, ...]]]] = {}
    # Validate core geometry fields; "frame" is image data and "frame_idx" is validated separately.
    keys_to_check = ("pred_keypoints_3d", "pred_keypoints_2d", "pred_vertices", "frame")
    for key in keys_to_check:
        shapes = {}
        for view, output in outputs.items():
            value = output.get(key)
            shapes[view] = value.shape if isinstance(value, np.ndarray) else None
        if len(set(shapes.values())) > 1:
            mismatched_shapes[key] = shapes

    frame_idx_map = {view: output.get("frame_idx") for view, output in outputs.items()}
    frame_idx_values = {val for val in frame_idx_map.values() if val is not None}
    frame_idx_mismatch = frame_idx_map if len(frame_idx_values) > 1 else {}
    frame_idx = None if frame_idx_mismatch else next(iter(frame_idx_values), None)

    has_missing = any(missing_keys[view] for view in missing_keys)
    if not has_missing and not mismatched_shapes and not frame_idx_mismatch:
        return None

    return {
        "frame_idx": frame_idx,
        "missing_keys": missing_keys,
        "shape_mismatch": mismatched_shapes,
        "frame_idx_mismatch": frame_idx_mismatch,
        "npz_paths": {view: str(path) for view, path in npz_paths.items()},
    }


MAPPING = {
    "night_high": "夜多い",
    "night_low": "夜少ない",
    "day_high": "昼多い",
    "day_low": "昼少ない",
}


def get_annotation_dict(file_path: str) -> dict:
    """
    JSONファイルからアノテーション情報を抽出し、
    { person_id: { env_name: { label: frame_num } } } の形式で返します。
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Annotation file not found: {file_path}")
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    master_dict = {}

    for item in data:
        video_path = item.get("video", "")
        file_name = os.path.basename(video_path)
        parts = file_name.split("_")

        # インデックスエラーを防ぐためのガード
        if len(parts) < 4:
            continue

        person = f"{parts[0]}_{parts[1]}"
        env_key = f"{parts[2]}_{parts[3]}"
        env_name = MAPPING.get(env_key, env_key)

        # 辞書の初期化を setdefault で簡潔に
        person_entry = master_dict.setdefault(person, {})

        # 必要なラベル情報を抽出
        frames = {"start": None, "mid": None, "end": None}
        for label_obj in item.get("videoLabels", []):
            labels = label_obj.get("timelinelabels", [])
            if not labels:
                continue

            label_name = labels[0]
            if label_name in frames:
                # ranges[0]['start'] が存在するか安全に取得
                ranges = label_obj.get("ranges", [{}])
                start = ranges[0].get("start")
                if start is not None:
                    # 标注是 24 fps 帧号，融合用的是相机帧号
                    frames[label_name] = annotation_to_camera_frame(start)

        person_entry[env_name] = frames

    return master_dict

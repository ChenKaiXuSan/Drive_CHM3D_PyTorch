#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/head3D_fuse/infer copy.py
Project: /workspace/code/head3D_fuse
Created Date: Tuesday February 3rd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday February 3rd 2026 9:16:54 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)
MIN_POINTS_FOR_ALIGNMENT = 3
MIN_POINTS_FOR_FIXED_ROTATION = 6
VALID_ALIGNMENT_METHODS = (
    "fixed_rotation",
    "procrustes",
    "procrustes_trimmed",
    "none",
)


def _normalize_keypoints(keypoints: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if keypoints is None:
        return None
    keypoints = np.asarray(keypoints)
    if keypoints.ndim == 3 and keypoints.shape[0] >= 1:
        return keypoints[0]
    return keypoints


def _valid_keypoints_mask(keypoints: np.ndarray, zero_eps: float) -> np.ndarray:
    finite = np.isfinite(keypoints).all(axis=-1)
    nonzero = np.linalg.norm(keypoints, axis=-1) >= zero_eps
    return finite & nonzero


def _estimate_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    allow_scale: bool,
    eps: float = 1e-8,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity transform mapping ``source`` onto ``target``.

    Points are ROW vectors, so the result is applied as
    ``scale * (source @ rot) + translation``. ``rot`` is therefore ``U @ D @ Vt``
    (the transpose of the column-vector Kabsch solution ``V @ D @ Ut``).
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    h_mat = source_centered.T @ target_centered
    u_mat, s_vals, vt_mat = np.linalg.svd(h_mat)
    # Reflection fix: keep a proper rotation (det = +1).
    sign = np.ones(3)
    if np.linalg.det(u_mat @ vt_mat) < 0:
        sign[-1] = -1.0
    rot = (u_mat * sign) @ vt_mat
    scale = 1.0
    if allow_scale:
        denom = float(np.sum(source_centered**2))
        scale = float(np.sum(s_vals * sign) / denom) if denom > eps else 1.0
    translation = target_mean - scale * (source_mean @ rot)
    return scale, rot, translation


def _chordal_mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Average rotation matrices (K, 3, 3) by projecting their sum onto SO(3)."""
    u_mat, _, vt_mat = np.linalg.svd(np.sum(rotations, axis=0))
    sign = np.ones(3)
    if np.linalg.det(u_mat @ vt_mat) < 0:
        sign[-1] = -1.0
    return (u_mat * sign) @ vt_mat


def rotation_angle_deg(rot: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rot) - 1.0) / 2.0, -1.0, 1.0))))


def _select_indices(
    mask: np.ndarray, keypoint_indices: Optional[Sequence[int]]
) -> np.ndarray:
    if keypoint_indices is None:
        return mask
    allowed = np.zeros_like(mask)
    allowed[list(keypoint_indices)] = True
    return mask & allowed


def estimate_fixed_view_rotations(
    keypoints_by_view: Dict[str, np.ndarray],
    reference: str = "front",
    zero_eps: float = 1e-6,
    keep_ratio: float = 0.5,
    min_points: int = MIN_POINTS_FOR_FIXED_ROTATION,
    keypoint_indices: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """Estimate ONE rotation per view that maps it into the reference camera frame.

    The cameras do not move during a recording, so the rotation from each view to
    the reference view is constant. Per-frame rotations are computed, the
    ``keep_ratio`` fraction with the smallest residual is kept, and they are
    averaged on SO(3).

    Args:
        keypoints_by_view: {view: (T, N, 3)} raw per-camera keypoints, same frames.
        reference: view whose camera frame is the common frame.

    Returns:
        rotations: {view: (3, 3)}, row-vector convention (``points @ rot``).
        info: {view: quality numbers} for logging/saving.
    """
    if reference not in keypoints_by_view:
        raise ValueError(f"Reference view '{reference}' not found in views.")
    ref_seq = np.asarray(keypoints_by_view[reference], dtype=np.float64)
    rotations: Dict[str, np.ndarray] = {reference: np.eye(3)}
    info: Dict[str, dict] = {}
    for view, seq in keypoints_by_view.items():
        if view == reference:
            continue
        seq = np.asarray(seq, dtype=np.float64)
        if seq.shape != ref_seq.shape:
            raise ValueError(
                f"View '{view}' has shape {seq.shape}, reference has {ref_seq.shape}"
            )
        candidates = []
        for src, ref in zip(seq, ref_seq):
            mask = _select_indices(
                _valid_keypoints_mask(src, zero_eps)
                & _valid_keypoints_mask(ref, zero_eps),
                keypoint_indices,
            )
            if mask.sum() < min_points:
                continue
            scale, rot, trans = _estimate_similarity_transform(src[mask], ref[mask], True)
            residual = scale * (src[mask] @ rot) + trans - ref[mask]
            candidates.append((float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))), rot))
        if not candidates:
            raise ValueError(
                f"Cannot estimate rotation for view '{view}': no frame has "
                f">={min_points} valid common keypoints."
            )
        candidates.sort(key=lambda item: item[0])
        kept = candidates[: max(1, int(len(candidates) * keep_ratio))]
        mean_rot = _chordal_mean_rotation(np.stack([rot for _, rot in kept]))
        deviation = [rotation_angle_deg(rot.T @ mean_rot) for _, rot in kept]
        rotations[view] = mean_rot
        info[view] = {
            "n_candidate_frames": len(candidates),
            "n_kept_frames": len(kept),
            "angle_to_reference_deg": rotation_angle_deg(mean_rot),
            "median_residual_m": float(np.median([res for res, _ in kept])),
            "median_deviation_deg": float(np.median(deviation)),
        }
    return rotations, info


def estimate_view_placements(
    keypoints_by_view: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    reference: str = "front",
    zero_eps: float = 1e-6,
    allow_scale: bool = True,
    keypoint_indices: Optional[Sequence[int]] = None,
) -> Dict[str, dict]:
    """Typical scale and offset that place each rotated view in the reference frame.

    ``points_in_reference ~= scale * (points @ rotation) + offset``. Medians over the
    frames where both views exist; only used for frames in which the reference view
    itself was not detected, so the other views can still be fused.
    """
    ref_seq = np.asarray(keypoints_by_view[reference], dtype=np.float64)
    placements: Dict[str, dict] = {reference: {"scale": 1.0, "offset": np.zeros(3)}}
    for view, seq in keypoints_by_view.items():
        if view == reference:
            continue
        scales, offsets = [], []
        for src, ref in zip(np.asarray(seq, dtype=np.float64), ref_seq):
            pair = _select_indices(
                _valid_keypoints_mask(src, zero_eps)
                & _valid_keypoints_mask(ref, zero_eps),
                keypoint_indices,
            )
            if pair.sum() < MIN_POINTS_FOR_ALIGNMENT:
                continue
            rotated = src[pair] @ rotations[view]
            src_center, ref_center = rotated.mean(axis=0), ref[pair].mean(axis=0)
            scale = 1.0
            denom = float(np.sum((rotated - src_center) ** 2))
            if allow_scale and denom > 1e-8:
                scale = float(np.sqrt(np.sum((ref[pair] - ref_center) ** 2) / denom))
            scales.append(scale)
            offsets.append(ref_center - scale * src_center)
        if scales:
            placements[view] = {
                "scale": float(np.median(scales)),
                "offset": np.median(np.stack(offsets), axis=0),
            }
    return placements


def align_view_with_fixed_rotation(
    reference: np.ndarray,
    source: np.ndarray,
    rotation: np.ndarray,
    zero_eps: float = 1e-6,
    allow_scale: bool = True,
    keypoint_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Rotate ``source`` into the reference frame, then match centroid (and scale).

    A view that cannot be placed (too few common points) becomes NaN for this frame,
    so it is ignored by the fusion instead of being mixed in unaligned.
    """
    ref = np.asarray(reference, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    src_valid = _valid_keypoints_mask(src, zero_eps)
    pair = _select_indices(
        src_valid & _valid_keypoints_mask(ref, zero_eps), keypoint_indices
    )
    if pair.sum() < MIN_POINTS_FOR_ALIGNMENT:
        return np.full_like(src, np.nan)
    rotated = src @ np.asarray(rotation, dtype=np.float64)
    src_center = rotated[pair].mean(axis=0)
    ref_center = ref[pair].mean(axis=0)
    scale = 1.0
    if allow_scale:
        denom = float(np.sum((rotated[pair] - src_center) ** 2))
        if denom > 1e-8:
            scale = float(np.sqrt(np.sum((ref[pair] - ref_center) ** 2) / denom))
    aligned = scale * (rotated - src_center) + ref_center
    aligned[~src_valid] = np.nan
    return aligned


def _align_keypoints_to_reference(
    reference: np.ndarray,
    source: np.ndarray,
    zero_eps: float,
    allow_scale: bool,
) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    ref_valid = _valid_keypoints_mask(ref, zero_eps)
    src_valid = _valid_keypoints_mask(src, zero_eps)
    pair_valid = ref_valid & src_valid
    if pair_valid.sum() < MIN_POINTS_FOR_ALIGNMENT:
        logger.warning(
            "Not enough valid joints for alignment (need >=%d, got %d).",
            MIN_POINTS_FOR_ALIGNMENT,
            pair_valid.sum(),
        )
        return source
    scale, rot, translation = _estimate_similarity_transform(
        src[pair_valid], ref[pair_valid], allow_scale
    )
    return _finalize_aligned_keypoints(src, source, src_valid, scale, rot, translation)


def _finalize_aligned_keypoints(
    source_points: np.ndarray,
    original_source: np.ndarray,
    valid_mask: np.ndarray,
    scale: float,
    rot: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    aligned = (scale * source_points @ rot) + translation
    aligned[~valid_mask] = original_source[~valid_mask]
    return aligned


def _select_trimmed_inliers(
    residuals: np.ndarray,
    valid_mask: np.ndarray,
    trim_ratio: float,
) -> np.ndarray:
    if trim_ratio <= 0:
        # trim_ratio <= 0 means no trimming; keep all valid points.
        return valid_mask
    if trim_ratio >= 1.0:
        raise ValueError(
            f"trim_ratio must be less than 1.0 (<=0 disables trimming), got {trim_ratio}"
        )
    valid_idx = np.flatnonzero(valid_mask)
    n_valid = valid_idx.size
    if n_valid == 0:
        return valid_mask
    raw_keep = int(np.ceil((1.0 - trim_ratio) * n_valid))
    if raw_keep < MIN_POINTS_FOR_ALIGNMENT:
        logger.warning(
            "trim_ratio keeps fewer than %d points; clamping to minimum.",
            MIN_POINTS_FOR_ALIGNMENT,
        )
    n_keep = min(n_valid, max(MIN_POINTS_FOR_ALIGNMENT, raw_keep))
    order = np.argsort(residuals[valid_idx])
    keep_idx = valid_idx[order[:n_keep]]
    trimmed = np.zeros_like(valid_mask, dtype=bool)
    trimmed[keep_idx] = True
    return trimmed


def _align_keypoints_trimmed(
    reference: np.ndarray,
    source: np.ndarray,
    zero_eps: float,
    allow_scale: bool,
    trim_ratio: float,
    max_iters: int,
) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    src = np.asarray(source, dtype=np.float64)
    ref_valid = _valid_keypoints_mask(ref, zero_eps)
    src_valid = _valid_keypoints_mask(src, zero_eps)
    pair_valid = ref_valid & src_valid
    if pair_valid.sum() < MIN_POINTS_FOR_ALIGNMENT:
        logger.warning(
            "Not enough valid joints for robust alignment (need >=%d, got %d).",
            MIN_POINTS_FOR_ALIGNMENT,
            pair_valid.sum(),
        )
        return source
    inlier_mask = pair_valid.copy()
    scale = 1.0
    rot = np.eye(3)
    translation = np.zeros(3)
    for _ in range(max_iters):
        if inlier_mask.sum() < MIN_POINTS_FOR_ALIGNMENT:
            break
        scale, rot, translation = _estimate_similarity_transform(
            src[inlier_mask], ref[inlier_mask], allow_scale
        )
        aligned = (scale * src @ rot) + translation
        residuals = np.linalg.norm(aligned - ref, axis=-1)
        new_inlier_mask = _select_trimmed_inliers(residuals, pair_valid, trim_ratio)
        if np.array_equal(new_inlier_mask, inlier_mask):
            break
        inlier_mask = new_inlier_mask
    if inlier_mask.sum() < MIN_POINTS_FOR_ALIGNMENT:
        logger.warning(
            "Robust alignment fell back to original source (inliers=%d, min=%d).",
            inlier_mask.sum(),
            MIN_POINTS_FOR_ALIGNMENT,
        )
        return source
    return _finalize_aligned_keypoints(src, source, src_valid, scale, rot, translation)


def _apply_view_transform(
    keypoints: Optional[np.ndarray],
    transform: Optional[Dict[str, np.ndarray]],
    mode: str,
) -> Optional[np.ndarray]:
    """
    Align keypoints to a common world coordinate using per-view extrinsics.

    Args:
        keypoints: (N, 3) 3D keypoints in the source coordinate system.
        transform: dict containing:
            - "R": (3, 3) rotation matrix.
            - "t": (3,) camera origin in world coordinates (same as C).
            - "t_wc": (3,) optional world->camera translation (OpenCV style).
              With OpenCV convention X_cam = R_wc @ X_world + t_wc and row-vector
              points, the camera origin is C = -(t_wc @ R_wc).
            - "C": (3,) optional camera origin in world (alias of "t").
        mode:
            - "world_to_camera": uses world->camera extrinsics (R_wc, t_wc) to align
              camera coordinates into world coordinates. With row-vector points,
              X_world = (X_cam - t_wc) @ R_wc. If "C" or "t" is provided instead of
              "t_wc", uses X_world = X_cam @ R_wc + C.
            - "camera_to_world": uses camera->world extrinsics (R_cw, t_cw) via
              X_world = X_cam @ R_cw.T + t_cw (camera origin in world).
    """
    if keypoints is None or transform is None:
        return keypoints
    if transform.get("R") is None:
        raise ValueError("transform['R'] is required for view alignment")
    R = np.asarray(transform.get("R"))
    if R.shape != (3, 3):
        raise ValueError(f"Expected rotation matrix with shape (3, 3), got {R.shape}")
    t = transform.get("t")
    if t is not None:
        t = np.asarray(t, dtype=np.float64).reshape(3)
    t_wc = transform.get("t_wc")
    if t_wc is not None:
        t_wc = np.asarray(t_wc, dtype=np.float64).reshape(3)
    camera_center = transform.get("C")
    if camera_center is not None:
        camera_center = np.asarray(camera_center, dtype=np.float64).reshape(3)
    if camera_center is None and t is not None:
        camera_center = t
    keypoints = np.asarray(keypoints, dtype=np.float64)
    if mode == "world_to_camera":
        if t_wc is None:
            if camera_center is None:
                raise ValueError("world_to_camera mode requires 't', 't_wc', or 'C'")
            return (keypoints @ R) + camera_center
        return (keypoints - t_wc) @ R
    if mode == "camera_to_world":
        if camera_center is None:
            raise ValueError("camera_to_world mode requires 't' or 'C' translation")
        return (keypoints @ R.T) + camera_center
    raise ValueError(
        f"transform_mode must be 'world_to_camera' or 'camera_to_world', got '{mode}'"
    )


def fuse_3view_keypoints(
    keypoints_by_view: Dict[str, np.ndarray],
    method: str = "median",
    zero_eps: float = 1e-6,
    fill_value: Optional[float] = None,
    view_transforms: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    transform_mode: str = "world_to_camera",
    alignment_method: str = "none",
    alignment_reference: Optional[str] = None,
    alignment_scale: bool = True,
    alignment_trim_ratio: float = 0.2,
    alignment_max_iters: int = 3,
    view_rotations: Optional[Dict[str, np.ndarray]] = None,
    alignment_keypoint_indices: Optional[Sequence[int]] = None,
    view_placements: Optional[Dict[str, dict]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fuse three-view 3D keypoints by ignoring invalid (near-zero) joints.

    Per-view keypoints live in their own camera frames, so they must be brought into
    one frame before fusing. The recommended mode is
    ``alignment_method="fixed_rotation"`` with ``alignment_reference="front"`` and
    ``view_rotations`` from :func:`estimate_fixed_view_rotations`. A missing view is
    passed as an all-NaN array; if the reference view itself is missing, the others are
    placed with ``view_placements`` from :func:`estimate_view_placements`.

    Optionally apply per-view extrinsics to align each view into a common
    world coordinate system before fusing. If no extrinsics are available,
    a Procrustes alignment (rotation/translation and optional scale) can be
    used to align each view to a reference view.

    Args:
        alignment_trim_ratio: fraction of points to trim as outliers (0.0 to <1.0).
            Values <= 0 keep all points. Only used when alignment_method is
            "procrustes_trimmed".
        alignment_max_iters: maximum iterations for trimmed alignment refinement.
            Only used when alignment_method is "procrustes_trimmed".
    """
    if fill_value is None:
        fill_value = np.nan
    if alignment_method not in VALID_ALIGNMENT_METHODS:
        raise ValueError(
            f"alignment_method must be one of {VALID_ALIGNMENT_METHODS}, "
            f"got '{alignment_method}'"
        )
    view_list = list(keypoints_by_view.keys())
    if view_transforms:
        keypoints_by_view = {
            view: _apply_view_transform(
                keypoints_by_view[view],
                view_transforms.get(view),
                transform_mode,
            )
            for view in view_list
        }
    elif alignment_method == "fixed_rotation":
        reference_view = alignment_reference or "front"
        if reference_view not in keypoints_by_view:
            raise ValueError(f"Reference view '{reference_view}' not found in views.")
        missing = [
            view
            for view in view_list
            if view != reference_view and view not in (view_rotations or {})
        ]
        if missing:
            raise ValueError(
                "alignment_method='fixed_rotation' needs view_rotations for "
                f"{missing}; call estimate_fixed_view_rotations() first."
            )
        reference = keypoints_by_view[reference_view]
        if _valid_keypoints_mask(np.asarray(reference, dtype=np.float64), zero_eps).any():
            keypoints_by_view = {
                view: (
                    reference
                    if view == reference_view
                    else align_view_with_fixed_rotation(
                        reference,
                        keypoints_by_view[view],
                        view_rotations[view],
                        zero_eps=zero_eps,
                        allow_scale=alignment_scale,
                        keypoint_indices=alignment_keypoint_indices,
                    )
                )
                for view in view_list
            }
        else:
            # 参考视角没检测到人：其余视角用固定旋转 + 该段典型的尺度/偏移放到参考坐标系
            placements = view_placements or {}
            aligned = {}
            for view in view_list:
                src = np.asarray(keypoints_by_view[view], dtype=np.float64)
                if view == reference_view or view not in placements:
                    aligned[view] = np.full_like(src, np.nan)
                    continue
                placed = (
                    placements[view]["scale"] * (src @ view_rotations[view])
                    + placements[view]["offset"]
                )
                placed[~_valid_keypoints_mask(src, zero_eps)] = np.nan
                aligned[view] = placed
            keypoints_by_view = aligned
    elif alignment_method in ("procrustes", "procrustes_trimmed"):
        reference_view = alignment_reference or view_list[0]
        if reference_view not in keypoints_by_view:
            raise ValueError(f"Reference view '{reference_view}' not found in views.")
        reference = keypoints_by_view[reference_view]
        keypoints_by_view = {
            view: (
                reference
                if view == reference_view
                else (
                    _align_keypoints_to_reference(
                        reference,
                        keypoints_by_view[view],
                        zero_eps=zero_eps,
                        allow_scale=alignment_scale,
                    )
                    if alignment_method == "procrustes"
                    else _align_keypoints_trimmed(
                        reference,
                        keypoints_by_view[view],
                        zero_eps=zero_eps,
                        allow_scale=alignment_scale,
                        trim_ratio=alignment_trim_ratio,
                        max_iters=alignment_max_iters,
                    )
                )
            )
            for view in view_list
        }
    stacked = np.stack([keypoints_by_view[view] for view in view_list], axis=0).astype(
        np.float64
    )
    finite = np.isfinite(stacked).all(axis=-1)
    nonzero = np.linalg.norm(stacked, axis=-1) >= zero_eps
    valid = finite & nonzero

    fused_mask = valid.any(axis=0)
    n_valid = valid.sum(axis=0).astype(np.int64)
    fused = np.full(stacked.shape[1:], fill_value, dtype=np.float64)

    if method == "first":
        for joint_idx in range(stacked.shape[1]):
            for view_idx in range(stacked.shape[0]):
                if valid[view_idx, joint_idx]:
                    fused[joint_idx] = stacked[view_idx, joint_idx]
                    break
    elif method in ("mean", "median"):
        reducer = np.nanmean if method == "mean" else np.nanmedian
        stacked_slice = stacked[:, fused_mask, :]
        valid_slice = valid[:, fused_mask]
        stacked_slice[~valid_slice] = np.nan
        fused[fused_mask] = reducer(stacked_slice, axis=0)
    else:
        raise ValueError(
            "method must be one of: 'mean', 'median', 'first' "
            f"(default from cfg.infer.fuse_method), got '{method}'"
        )

    return fused, fused_mask, n_valid

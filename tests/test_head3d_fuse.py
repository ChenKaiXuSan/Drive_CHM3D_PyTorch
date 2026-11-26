import numpy as np
import pytest

from head3D_fuse.fuse.fuse import (
    _estimate_similarity_transform,
    align_view_with_fixed_rotation,
    estimate_fixed_view_rotations,
    estimate_view_placements,
)
from head3D_fuse.load import assemble_view_npz_paths, get_annotation_dict
from head3D_fuse.infer import fuse_3view_keypoints


def _rot(axis, degrees):
    """Row-vector rotation matrix about 'x', 'y' or 'z'."""
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    col = {
        "x": [[1, 0, 0], [0, c, -s], [0, s, c]],
        "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
    }[axis]
    return np.array(col, dtype=float).T


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("degrees", [-120, -40, 15, 40, 90, 179])
def test_similarity_transform_recovers_rotation(axis, degrees):
    rng = np.random.default_rng(0)
    src = rng.normal(size=(30, 3))
    rot_true, scale_true, t_true = _rot(axis, degrees), 1.3, np.array([0.1, -0.2, 0.5])
    ref = scale_true * (src @ rot_true) + t_true

    scale, rot, t = _estimate_similarity_transform(src, ref, allow_scale=True)

    assert np.allclose(rot, rot_true, atol=1e-8)
    assert np.isclose(scale, scale_true)
    assert np.allclose(scale * (src @ rot) + t, ref, atol=1e-8)


def test_similarity_transform_never_returns_reflection():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(30, 3))
    _, rot, _ = _estimate_similarity_transform(
        src, src * np.array([1.0, 1.0, -1.0]), allow_scale=True
    )
    assert np.isclose(np.linalg.det(rot), 1.0)


def _three_view_sequence(n_frames=40, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    body = rng.normal(scale=0.3, size=(70, 3))
    front = np.stack(
        [body @ _rot("y", 20 * np.sin(i / 5)) + [0, 0, 0.01 * i] for i in range(n_frames)]
    )
    to_front = {"left": _rot("y", 40), "right": _rot("y", -35) @ _rot("x", 5)}
    views = {"front": front}
    for view, rot in to_front.items():
        # front = view @ rot  =>  view = front @ rot.T (plus own offset and scale)
        views[view] = 0.9 * (front @ rot.T) + [0.3, 0.1, -0.2]
    views = {v: k + rng.normal(scale=noise, size=k.shape) for v, k in views.items()}
    return views, to_front


def test_fixed_rotation_recovers_view_rotations_and_fuses_to_front():
    views, to_front = _three_view_sequence()
    rotations, info = estimate_fixed_view_rotations(views, reference="front")

    assert np.allclose(rotations["front"], np.eye(3))
    for view, rot in to_front.items():
        assert np.allclose(rotations[view], rot, atol=1e-8)
        assert info[view]["median_residual_m"] < 1e-8

    for i in (0, 17, 39):
        fused, mask, n_valid = fuse_3view_keypoints(
            {v: k[i] for v, k in views.items()},
            alignment_method="fixed_rotation",
            alignment_reference="front",
            view_rotations=rotations,
        )
        assert np.allclose(fused, views["front"][i], atol=1e-8)
        assert mask.all() and (n_valid == 3).all()


def test_fixed_rotation_is_robust_to_bad_frames():
    views, to_front = _three_view_sequence(noise=0.005)
    rng = np.random.default_rng(3)
    views["left"][::4] = rng.normal(size=views["left"][::4].shape)  # 25% broken frames
    rotations, _ = estimate_fixed_view_rotations(views, reference="front")
    cos = (np.trace(rotations["left"].T @ to_front["left"]) - 1) / 2
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))) < 1.0


def test_fixed_rotation_requires_rotations():
    views, _ = _three_view_sequence(n_frames=1)
    with pytest.raises(ValueError, match="view_rotations"):
        fuse_3view_keypoints(
            {v: k[0] for v, k in views.items()},
            alignment_method="fixed_rotation",
            alignment_reference="front",
        )


def test_fixed_rotation_drops_view_that_cannot_be_placed():
    views, to_front = _three_view_sequence(n_frames=1)
    frame = {v: k[0].copy() for v, k in views.items()}
    frame["left"][2:] = np.nan  # only 2 valid points: cannot be aligned
    aligned = align_view_with_fixed_rotation(
        frame["front"], frame["left"], to_front["left"]
    )
    assert np.isnan(aligned).all()
    fused, _, n_valid = fuse_3view_keypoints(
        frame,
        alignment_method="fixed_rotation",
        alignment_reference="front",
        view_rotations={"front": np.eye(3), **to_front},
    )
    assert np.allclose(fused, frame["front"], atol=1e-8)
    assert (n_valid == 2).all()


def test_missing_front_is_filled_from_other_views():
    views, _ = _three_view_sequence()
    rotations, _ = estimate_fixed_view_rotations(views, reference="front")
    placements = estimate_view_placements(views, rotations, reference="front")
    for i in (0, 11, 39):
        frame = {v: k[i].copy() for v, k in views.items()}
        frame["front"] = np.full_like(frame["front"], np.nan)
        fused, mask, n_valid = fuse_3view_keypoints(
            frame,
            alignment_method="fixed_rotation",
            alignment_reference="front",
            view_rotations=rotations,
            view_placements=placements,
        )
        assert np.allclose(fused, views["front"][i], atol=1e-6)
        assert mask.all() and (n_valid == 2).all()


def test_missing_front_without_placements_gives_no_result():
    views, to_front = _three_view_sequence(n_frames=1)
    frame = {v: k[0].copy() for v, k in views.items()}
    frame["front"][:] = np.nan
    fused, mask, _ = fuse_3view_keypoints(
        frame,
        alignment_method="fixed_rotation",
        alignment_reference="front",
        view_rotations={"front": np.eye(3), **to_front},
    )
    assert np.isnan(fused).all() and not mask.any()


def test_assemble_frames_with_min_views(tmp_path):
    env = tmp_path / "01" / "昼多い"
    have = {"front": [1, 2], "left": [1, 2, 3], "right": [1, 3, 4]}
    for view, frames in have.items():
        (env / view).mkdir(parents=True)
        for f in frames:
            (env / view / f"{f:06d}_sam3d_body.npz").touch()
    strict, _ = assemble_view_npz_paths(env, list(have))
    assert [t.frame_idx for t in strict] == [1]
    loose, _ = assemble_view_npz_paths(env, list(have), min_views=2)
    assert [t.frame_idx for t in loose] == [1, 2, 3]
    assert set(loose[2].npz_paths) == {"left", "right"}


def test_start_mid_end_are_camera_frames(tmp_path):
    import json

    path = tmp_path / "mini.json"
    path.write_text(
        json.dumps(
            [
                {
                    "video": "/x/person_01_day_high_h265.mp4",
                    "videoLabels": [
                        {"ranges": [{"start": s, "end": s}], "timelinelabels": [n]}
                        for n, s in (("start", 366), ("mid", 5218), ("end", 12155))
                    ],
                }
            ]
        )
    )
    expected = {"start": 457, "mid": 6516, "end": 15179}
    assert get_annotation_dict(str(path))["person_01"]["昼多い"] == expected


def test_fuse_3view_keypoints_mean():
    keypoints = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    keypoints_by_view = {
        "front": keypoints,
        "left": keypoints * 2,
        "right": keypoints * 3,
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view, method="mean"
    )

    assert np.allclose(fused[0], np.array([2.0, 4.0, 6.0]))
    assert np.isnan(fused[1]).all()
    assert fused_mask.tolist() == [True, False]
    assert n_valid.tolist() == [3, 0]


def test_fuse_3view_keypoints_first():
    keypoints_by_view = {
        "front": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        "left": np.array([[2.0, 2.0, 2.0], [0.0, 0.0, 0.0]]),
        "right": np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]),
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view, method="first"
    )

    assert np.allclose(fused[0], np.array([2.0, 2.0, 2.0]))
    assert np.allclose(fused[1], np.array([1.0, 1.0, 1.0]))
    assert fused_mask.tolist() == [True, True]
    assert n_valid.tolist() == [2, 2]


def test_fuse_3view_keypoints_median_and_nan():
    keypoints_by_view = {
        "front": np.array([[1.0, 1.0, 1.0], [np.nan, 0.0, 0.0]]),
        "left": np.array([[3.0, 3.0, 3.0], [2.0, 2.0, 2.0]]),
        "right": np.array([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]]),
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view, method="median"
    )

    assert np.allclose(fused[0], np.array([3.0, 3.0, 3.0]))
    assert np.allclose(fused[1], np.array([2.0, 2.0, 2.0]))
    assert fused_mask.tolist() == [True, True]
    assert n_valid.tolist() == [3, 1]


def test_fuse_3view_keypoints_with_extrinsic_alignment():
    keypoints_by_view = {
        "front": np.array([[1.0, 1.0, 1.0]]),
        "left": np.array([[2.0, 1.0, 1.0]]),
        "right": np.array([[0.0, 1.0, 1.0]]),
    }
    view_transforms = {
        "front": {"R": np.eye(3), "t_wc": np.zeros(3)},
        "left": {"R": np.eye(3), "t_wc": np.array([1.0, 0.0, 0.0])},
        "right": {"R": np.eye(3), "t_wc": np.array([-1.0, 0.0, 0.0])},
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        view_transforms=view_transforms,
        transform_mode="world_to_camera",
    )

    assert np.allclose(fused[0], np.array([1.0, 1.0, 1.0]))
    assert fused_mask.tolist() == [True]
    assert n_valid.tolist() == [3]


def test_fuse_3view_keypoints_with_procrustes_alignment():
    ref = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotation_matrix = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation_vector = np.array([2.0, -1.0, 0.0])
    transformed = (ref @ rotation_matrix) + translation_vector
    keypoints_by_view = {
        "front": ref,
        "left": transformed,
        "right": transformed,
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        alignment_method="procrustes",
        alignment_reference="front",
        alignment_scale=False,
    )

    assert np.allclose(fused, ref, atol=1e-6)
    assert fused_mask.tolist() == [True, True, True]
    assert n_valid.tolist() == [3, 3, 3]


def test_fuse_3view_keypoints_with_trimmed_alignment():
    ref = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    rotation_matrix = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation_vector = np.array([2.0, -1.0, 0.0])
    transformed = (ref @ rotation_matrix) + translation_vector
    transformed[0] = np.array([10.0, 10.0, 0.0])  # inject a strong outlier
    keypoints_by_view = {
        "front": ref,
        "left": transformed,
        "right": transformed,
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        alignment_method="procrustes_trimmed",
        alignment_reference="front",
        alignment_scale=False,
        alignment_trim_ratio=0.2,
        alignment_max_iters=5,
    )

    # Joint 0 is the injected outlier itself; alignment can only restore the inliers.
    assert np.allclose(fused[1:], ref[1:], atol=1e-6)
    assert fused_mask.tolist() == [True, True, True, True, True]
    assert n_valid.tolist() == [3, 3, 3, 3, 3]


def test_fuse_3view_keypoints_trimmed_alignment_fallback():
    ref = np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.0, 0.0]])
    keypoints_by_view = {
        "front": ref,
        "left": ref + 1.0,
        "right": ref - 1.0,
    }

    fused_base, fused_mask_base, n_valid_base = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
    )

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        alignment_method="procrustes_trimmed",
        alignment_reference="front",
        alignment_scale=False,
        alignment_trim_ratio=0.99,
        alignment_max_iters=5,
    )

    assert np.allclose(fused, fused_base)
    assert fused_mask.tolist() == [True, True, True]
    assert n_valid.tolist() == [3, 3, 3]


def test_fuse_3view_keypoints_trimmed_alignment_moderate():
    ref = np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ]
    )
    rotation_matrix = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translation_vector = np.array([2.0, -1.0, 0.0])
    transformed = (ref @ rotation_matrix) + translation_vector
    transformed[0] = np.array([8.0, 8.0, 0.0])  # inject a moderate outlier
    keypoints_by_view = {
        "front": ref,
        "left": transformed,
        "right": transformed,
    }

    fused, fused_mask, n_valid = fuse_3view_keypoints(
        keypoints_by_view,
        method="mean",
        alignment_method="procrustes_trimmed",
        alignment_reference="front",
        alignment_scale=False,
        alignment_trim_ratio=0.5,
        alignment_max_iters=5,
    )

    # Joint 0 is the injected outlier itself; alignment can only restore the inliers.
    assert np.allclose(fused[1:], ref[1:], atol=1e-6)
    assert fused_mask.tolist() == [True, True, True, True, True]
    assert n_valid.tolist() == [3, 3, 3, 3, 3]

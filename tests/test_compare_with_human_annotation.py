import json

import numpy as np
import pytest

from compare_with_human_annotation.angle_calculator import (
    LABEL_DIRECTION_MAP,
    axis_direction_matched,
    calculate_head_angles,
    circular_mean_deg,
    classify_label,
    estimate_front_yaw,
    estimate_stable_front_baseline,
    wrap_angle_deg,
)
from compare_with_human_annotation.head_pose_analyzer import HeadPoseAnalyzer
from common.frame_rate import annotation_to_camera_frame
from compare_with_human_annotation.load import (
    HeadMovementLabel,
    load_annotations_by_annotator,
    load_majority_voted_annotations,
)
from head_movement_analysis import angle_calculator as hma
from head_movement_analysis import load as hma_load


def _head(yaw_right=0.0, pitch_up=0.0, camera_yaw=0.0):
    """Head keypoints in a camera frame (X right, Y down, Z away from camera).

    yaw_right > 0: driver turns to the driver's own right; pitch_up > 0: looks up.
    """
    # 驾驶员面对相机：鼻子朝 -Z，驾驶员左侧在图像右侧(+X)
    pts = {
        "nose": [0.0, 0.0, -0.10],
        "left_eye": [0.03, -0.03, -0.08],
        "right_eye": [-0.03, -0.03, -0.08],
        "left_ear": [0.07, 0.0, 0.0],
        "right_ear": [-0.07, 0.0, 0.0],
    }
    a, b = np.radians(yaw_right + camera_yaw), np.radians(pitch_up)
    # 抬头：鼻子(-Z)向上(-Y)转，绕 X 轴
    rx = np.array([[1, 0, 0], [0, np.cos(b), np.sin(b)], [0, -np.sin(b), np.cos(b)]])
    # 向驾驶员右侧：鼻子(-Z)向 -X 转，绕 Y 轴
    ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    out = {k: ry @ rx @ np.array(v) + np.array([0.0, -0.6, 0.0]) for k, v in pts.items()}
    out["left_shoulder"] = np.array([0.2, -0.4, 0.0])
    out["right_shoulder"] = np.array([-0.2, -0.4, 0.0])
    out["neck"] = np.array([0.0, -0.45, 0.0])
    return out


@pytest.mark.parametrize("yaw", [-60, -20, 0, 20, 60])
@pytest.mark.parametrize("pitch", [-25, 0, 25])
def test_head_angles_follow_driver_convention(yaw, pitch):
    got_pitch, got_yaw = calculate_head_angles(_head(yaw, pitch))
    assert got_yaw == pytest.approx(yaw, abs=1e-6)
    assert got_pitch == pytest.approx(pitch, abs=1e-6)
    hma_pitch, hma_yaw, _ = hma.calculate_head_angles(_head(yaw, pitch))
    assert hma_yaw == pytest.approx(yaw, abs=1e-6)
    assert hma_pitch == pytest.approx(pitch, abs=1e-6)


def test_roll_is_zero_for_level_head():
    assert hma.calculate_head_angles(_head())[2] == pytest.approx(0.0, abs=1e-6)


def test_annotation_frames_are_converted_to_camera_frames():
    assert annotation_to_camera_frame(0) == 0
    assert annotation_to_camera_frame(24) == 30
    assert annotation_to_camera_frame(10000) == 12488


def _task(video, annotations):
    return {
        "data": {"video": f"/x/{video}.mp4"},
        "annotations": [
            {
                "completed_by": who,
                "result": [
                    {
                        "type": "timelinelabels",
                        "value": {"ranges": [{"start": s, "end": e}], "timelinelabels": [lab]},
                    }
                    for lab, s, e in labels
                ],
            }
            for who, labels in annotations
        ],
    }


def test_annotators_are_identified_by_completed_by(tmp_path):
    # 同一位置在两个视频里是不同的人；第二个视频缺少标注者 4
    data = [
        _task("person_01_day_high_h265", [(5, [("left", 24, 48)]), (3, [("up", 0, 24)]), (4, [("down", 0, 24)])]),
        _task("person_02_day_high_h265", [(3, [("up", 0, 24)]), (5, [("left", 24, 48)])]),
    ]
    path = tmp_path / "full.json"
    path.write_text(json.dumps(data))
    out = load_annotations_by_annotator(path)
    for video in ("01_day_high", "02_day_high"):
        assert len(out[video]) == 3
        assert [x.label for x in out[video][0]] == ["up"]  # 标注者 3
        assert [x.label for x in out[video][2]] == ["left"]  # 标注者 5
    assert [x.label for x in out["01_day_high"][1]] == ["down"]
    assert out["02_day_high"][1] == []
    left = out["01_day_high"][2][0]
    assert (left.start_frame, left.end_frame) == (30, 60)
    assert hma_load.load_head_movement_annotations(path)["01_day_high"][0].start_frame == 30


def test_majority_vote_uses_camera_frames(tmp_path):
    path = tmp_path / "full.json"
    path.write_text(json.dumps([_task("person_01_day_high_h265", [(i, [("left_up", 240, 480)]) for i in (3, 4, 5)])]))
    (label,) = load_majority_voted_annotations(path)["01_day_high"]
    assert (label.label, label.start_frame, label.end_frame) == ("left_up", 300, 599)


def test_front_baseline_is_most_common_direction_not_zero():
    rng = np.random.default_rng(0)
    yaw = np.r_[rng.normal(35, 1.0, 800), rng.normal(0, 8, 150), rng.normal(70, 5, 50)]
    pitch = np.r_[rng.normal(-12, 1.0, 800), rng.normal(10, 5, 200)]
    assert estimate_front_yaw(yaw) == pytest.approx(35, abs=1.0)
    baseline_pitch, selected, _ = estimate_stable_front_baseline(pitch, yaw)
    assert circular_mean_deg(yaw[selected]) == pytest.approx(35, abs=1.5)
    assert baseline_pitch == pytest.approx(-12, abs=1.0)


def test_angles_wrap_around():
    assert wrap_angle_deg(350) == pytest.approx(-10)
    assert wrap_angle_deg(-190) == pytest.approx(170)
    assert abs(wrap_angle_deg(circular_mean_deg([179, -179]) - 180)) < 1e-6
    out = HeadPoseAnalyzer._apply_baseline({"pitch": 0, "yaw": -175}, {"pitch": 0, "yaw": 175})
    assert out["yaw"] == pytest.approx(10)


def test_diagonal_labels_are_evaluated_on_both_axes():
    assert LABEL_DIRECTION_MAP["left_down"] == (-1, -1)
    assert classify_label(-20, 30, 10) == "right_down"
    analyzer = HeadPoseAnalyzer({"v": [HeadMovementLabel(0, 10, "left_up")]})

    def run(pitch, yaw):
        return analyzer.compare_with_annotations("v", 5, {"pitch": pitch, "yaw": yaw}, threshold_deg=10)["matches"][0]

    good, half = run(20, -20), run(0, -20)
    assert good["yaw_axis_active"] and good["pitch_axis_active"]
    assert axis_direction_matched(good)
    assert half["yaw_match"] and not half["pitch_match"]
    assert not axis_direction_matched(half)

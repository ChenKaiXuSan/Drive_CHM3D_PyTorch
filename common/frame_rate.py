"""标注帧号与相机帧号的换算（全仓库唯一定义）。

Label Studio 导出的帧号(方向标注和 start/mid/end)按 24 fps 计，相机视频以及
SAM3D / 融合结果的帧号是 29.97 fps，两者必须换算后才能一起使用。
"""

ANNOTATION_FPS = 24.0
CAMERA_FPS = 29.97


def annotation_to_camera_frame(frame: float) -> int:
    """把标注帧号(24 fps)换算为相机帧号(29.97 fps)。"""
    return int(round(float(frame) * CAMERA_FPS / ANNOTATION_FPS))

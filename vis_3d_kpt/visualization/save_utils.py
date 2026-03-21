#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def save_figure(image: Any, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    image.savefig(save_path, dpi=300, facecolor="white", edgecolor="white")
    # close matplotlib figure
    try:
        import matplotlib.pyplot as plt

        plt.close(image)
    except Exception:
        pass


def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("Got None image")

    img = np.asarray(img)

    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        elif img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported channel number: {img.shape}")
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def merge_frame_to_video(save_path: Path, flag: str, fps: int = 30) -> Path:
    frame_dir = save_path / flag
    out_path = save_path / "video"
    out_path.mkdir(exist_ok=True, parents=True)

    frames = sorted(frame_dir.glob("*"), key=lambda x: int(x.stem.split("_")[0]))
    if not frames:
        raise RuntimeError(f"No frames found in {frame_dir}")

    output_file = out_path / f"{flag}.mp4"

    first_img = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
    first_img = _ensure_rgb_uint8(first_img)
    H, W = first_img.shape[:2]

    writer = None
    for codec in ("avc1", "H264", "X264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate = cv2.VideoWriter(str(output_file), fourcc, float(fps), (W, H))
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"Failed to open VideoWriter for {output_file}")

    writer.write(first_img)

    for f in frames[1:]:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        img = _ensure_rgb_uint8(img)

        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        writer.write(img)

    writer.release()
    return output_file

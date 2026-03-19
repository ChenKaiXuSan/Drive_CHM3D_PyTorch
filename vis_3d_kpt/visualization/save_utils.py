#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio_v2
import numpy as np
from PIL import Image


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

    first_img = imageio_v2.imread(frames[0])
    first_img = _ensure_rgb_uint8(first_img)
    H, W = first_img.shape[:2]

    # use FFMPEG backend
    writer = imageio_v2.get_writer(output_file, fps=fps, format="FFMPEG", codec="libx264")

    writer.append_data(first_img)

    for f in frames[1:]:
        img = imageio_v2.imread(f)
        img = _ensure_rgb_uint8(img)

        if img.shape[0] != H or img.shape[1] != W:
            img_pil = Image.fromarray(img)
            # PIL resampling compatibility
            try:
                resample = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
            except Exception:
                resample = Image.BILINEAR
            img_pil = img_pil.resize((W, H), resample)
            img = np.asarray(img_pil)

        writer.append_data(img)

    writer.close()
    return output_file

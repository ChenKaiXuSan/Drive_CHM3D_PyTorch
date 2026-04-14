#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot axis-wise confusion matrices from compare_with_human_annotation result files.

This script reads detailed `result.json` files that contain `comparisons` and
builds confusion matrices for:
1) Pitch axis (true up/down vs predicted down/neutral/up)
2) Yaw axis (true left/right vs predicted left/neutral/right)

The matrices follow the axis-evaluation semantics used by the project:
- Pitch matrix only includes matches where `pitch_axis_active` is True.
- Yaw matrix only includes matches where `yaw_axis_active` is True.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _dir_from_value(value: float, threshold: float) -> int:
    if value > threshold:
        return 1
    if value < -threshold:
        return -1
    return 0


def _collect_files(root: Path, threshold_deg: str, source: str, annotation_mode: str) -> List[Path]:
    threshold_dir = root / f"threshold_{threshold_deg}deg"
    if source == "single_view":
        pattern = threshold_dir / "single_view" / annotation_mode / "*" / "*" / "*" / "result.json"
    elif source == "fused_raw":
        pattern = threshold_dir / "fused" / "raw" / annotation_mode / "*" / "*" / "result.json"
    elif source == "fused_smoothed":
        pattern = threshold_dir / "fused" / "smoothed" / annotation_mode / "*" / "*" / "result.json"
    else:
        raise ValueError(f"Unsupported source: {source}")
    return sorted(pattern.parent.glob(pattern.name)) if "*" not in str(pattern) else sorted(Path().glob(str(pattern)))


def _safe_collect_files(root: Path, threshold_deg: str, source: str, annotation_mode: str) -> List[Path]:
    threshold_dir = root / f"threshold_{threshold_deg}deg"
    if source == "single_view":
        return sorted(threshold_dir.glob(f"single_view/{annotation_mode}/*/*/*/result.json"))
    if source == "fused_raw":
        return sorted(threshold_dir.glob(f"fused/raw/{annotation_mode}/*/*/result.json"))
    if source == "fused_smoothed":
        return sorted(threshold_dir.glob(f"fused/smoothed/{annotation_mode}/*/*/result.json"))
    raise ValueError(f"Unsupported source: {source}")


def _build_matrices(files: List[Path], threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    # pitch: rows true [-1, 1], cols pred [-1, 0, 1]
    pitch = np.zeros((2, 3), dtype=np.int64)
    # yaw: rows true [-1, 1], cols pred [-1, 0, 1]
    yaw = np.zeros((2, 3), dtype=np.int64)

    pitch_row = {-1: 0, 1: 1}
    yaw_row = {-1: 0, 1: 1}
    col = {-1: 0, 0: 1, 1: 2}

    used_files = 0
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        comparisons = data.get("comparisons", {})
        if not isinstance(comparisons, dict) or not comparisons:
            continue
        used_files += 1

        for comp in comparisons.values():
            for m in comp.get("matches", []):
                expected_pitch = int(m.get("expected_pitch", 0))
                expected_yaw = int(m.get("expected_yaw", 0))
                pred_pitch = _dir_from_value(float(m.get("pitch_value", 0.0)), threshold)
                pred_yaw = _dir_from_value(float(m.get("yaw_value", 0.0)), threshold)

                if m.get("pitch_axis_active", False) and expected_pitch in pitch_row:
                    pitch[pitch_row[expected_pitch], col[pred_pitch]] += 1

                if m.get("yaw_axis_active", False) and expected_yaw in yaw_row:
                    yaw[yaw_row[expected_yaw], col[pred_yaw]] += 1

    return pitch, yaw, used_files


def _plot_matrix(ax, mat: np.ndarray, row_labels: List[str], col_labels: List[str], title: str) -> None:
    im = ax.imshow(mat, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    vmax = max(int(mat.max()), 1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = int(mat[i, j])
            color = "white" if value > vmax * 0.5 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot axis-wise confusion matrices")
    parser.add_argument("--root", default="/workspace/data/compare_with_human_annotation_results")
    parser.add_argument("--threshold", default="10", help="e.g. 0, 5, 10, 15, 20")
    parser.add_argument(
        "--source",
        default="single_view",
        choices=["single_view", "fused_raw", "fused_smoothed"],
    )
    parser.add_argument("--annotation-mode", default="majority", choices=["majority", "by_annotator"])
    parser.add_argument("--output-dir", default="/workspace/data/compare_with_human_annotation_results/plots")
    args = parser.parse_args()

    root = Path(args.root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _safe_collect_files(root, args.threshold, args.source, args.annotation_mode)
    if not files:
        raise SystemExit("No result.json files found for the given settings.")

    threshold_value = float(args.threshold)
    pitch_mat, yaw_mat, used_files = _build_matrices(files, threshold_value)
    if used_files == 0:
        raise SystemExit(
            "Found result.json files, but none contains detailed comparisons. "
            "Please rerun with detailed outputs enabled."
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
    _plot_matrix(
        axes[0],
        pitch_mat,
        row_labels=["down (-1)", "up (+1)"],
        col_labels=["down (-1)", "neutral (0)", "up (+1)"],
        title="Pitch Axis Confusion",
    )
    _plot_matrix(
        axes[1],
        yaw_mat,
        row_labels=["left (-1)", "right (+1)"],
        col_labels=["left (-1)", "neutral (0)", "right (+1)"],
        title="Yaw Axis Confusion",
    )

    fig.suptitle(
        f"Axis Confusion | threshold={args.threshold}deg | source={args.source} | "
        f"mode={args.annotation_mode} | files={used_files}",
        fontsize=11,
    )
    fig.tight_layout()

    out_png = output_dir / f"axis_confusion_threshold_{args.threshold}deg_{args.source}_{args.annotation_mode}.png"
    fig.savefig(out_png)

    out_npz = output_dir / f"axis_confusion_threshold_{args.threshold}deg_{args.source}_{args.annotation_mode}.npz"
    np.savez_compressed(out_npz, pitch=pitch_mat, yaw=yaw_mat)

    print(f"Saved figure: {out_png}")
    print(f"Saved counts: {out_npz}")
    print("Pitch matrix (rows=true down/up, cols=pred down/neutral/up):")
    print(pitch_mat)
    print("Yaw matrix (rows=true left/right, cols=pred left/neutral/right):")
    print(yaw_mat)


if __name__ == "__main__":
    main()

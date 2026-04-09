#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""compare_with_human_annotation 模块入口（Hydra 版本）。"""

import hydra
from omegaconf import DictConfig

from .batch_run import (
    run_batch_comparison_by_annotator,
    run_batch_comparison_majority_vote,
)
from .run import (
    run_comparison,
    run_single_view_comparison,
    run_single_view_comparison_all,
)


def _dispatch(cfg: DictConfig) -> None:
    mode = str(cfg.run.mode).lower()
    annotation_mode = str(cfg.run.annotation_mode).lower()
    threshold = float(cfg.run.threshold)
    person_id = cfg.run.person_id
    env_jp = cfg.run.env_jp
    view = str(cfg.run.view)
    start_frame = cfg.run.start_frame
    end_frame = cfg.run.end_frame

    if annotation_mode not in {"majority", "by_annotator"}:
        raise ValueError(f"Unsupported run.annotation_mode: {cfg.run.annotation_mode}")

    if mode == "fused":
        if person_id is None and env_jp is None:
            if annotation_mode == "majority":
                run_batch_comparison_majority_vote(threshold_deg=threshold, cfg=cfg)
            else:
                run_batch_comparison_by_annotator(threshold_deg=threshold, cfg=cfg)
            return

        run_comparison(
            person_id=person_id,
            env_jp=env_jp,
            annotation_mode=annotation_mode,
            start_frame=start_frame,
            end_frame=end_frame,
            threshold=threshold,
            cfg=cfg,
        )
        return

    if mode == "single_view":
        if person_id is None and env_jp is None:
            run_single_view_comparison_all(
                view=view,
                annotation_mode=annotation_mode,
                threshold=threshold,
                cfg=cfg,
            )
            return

        run_single_view_comparison(
            person_id=person_id,
            env_jp=env_jp,
            view=view,
            annotation_mode=annotation_mode,
            threshold=threshold,
            start_frame=start_frame,
            end_frame=end_frame,
            cfg=cfg,
        )
        return

    raise ValueError(f"Unsupported run.mode: {cfg.run.mode}")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="compare_with_human_annotation",
)
def main(cfg: DictConfig) -> None:
    _dispatch(cfg)


if __name__ == "__main__":
    main()

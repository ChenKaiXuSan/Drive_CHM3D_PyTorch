#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""补跑每段视频末尾缺失的 SAM 3D Body 推理。

旧版把 24 fps 的标注帧号当作相机帧号，推理在 end 标记(24 fps 帧号)处提前停止。
本脚本只补 [旧结束帧, 换算后的 end] 这一段，保存格式与 SAM3Dbody.main 相同，可随时中断续跑。

用法:
    python -m SAM3Dbody.fill_missing --ckpt <sam-3d-body-dinov3 目录> --gpus 0 1 2 3 --workers-per-gpu 2
"""

import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path

VIEWS = ("front", "left", "right")
NONE_FILE = "none_detected_frames_fill.txt"


def build_tasks(args):
    """返回 [(person, env, view, first, last)]，帧号为相机帧号，含两端。"""
    import cv2

    from common.frame_rate import annotation_to_camera_frame
    from .load import MAPPING

    tasks = []
    for item in json.loads(Path(args.marks).read_text()):
        parts = os.path.basename(item["video"]).split("_")
        person, env = parts[1], MAPPING[f"{parts[2]}_{parts[3]}"]
        marks = {
            lab["timelinelabels"][0]: lab["ranges"][0]["start"]
            for lab in item["videoLabels"]
        }
        for view in VIEWS:
            video = Path(args.video_root) / person / env / f"{view}.mp4"
            cap = cv2.VideoCapture(str(video))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            first = int(marks["end"])  # 旧推理的 range(start, end) 不含 end
            last = min(annotation_to_camera_frame(marks["end"]), n_frames - 1)
            if last >= first:
                tasks.append((person, env, view, first, last))
    return sorted(tasks)


def pending_frames(args, person, env, view, first, last):
    from .save import get_frame_path

    out_dir = Path(args.result_root) / person / env / view
    none_file = out_dir / NONE_FILE
    none = set(map(int, none_file.read_text().split())) if none_file.exists() else set()
    frames = [
        i
        for i in range(first, last + 1)
        if i not in none and not get_frame_path(out_dir, i).exists()
    ]
    return out_dir, frames


def worker(gpu, rank, queue, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "4"
    import cv2
    import torch
    from omegaconf import OmegaConf

    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    from .infer import select_best_person, setup_sam_3d_body
    from .save import save_frame

    cfg = OmegaConf.create(
        {
            "model": {
                "checkpoint_path": f"{args.ckpt}/model.ckpt",
                "mhr_path": f"{args.ckpt}/assets/mhr_model.pt",
                "fov_name": "moge2",
            },
            "infer": {"gpu": 0},
        }
    )
    with contextlib.redirect_stdout(io.StringIO()):
        estimator = setup_sam_3d_body(cfg)
    torch.cuda.empty_cache = lambda: None  # 每帧清缓存只会拖慢，不影响结果
    if args.fp16_detector:
        detect = estimator.detector.run_human_detection

        def detect_fp16(*a, **k):
            with torch.autocast("cuda", dtype=torch.float16):
                return detect(*a, **k)

        estimator.detector.run_human_detection = detect_fp16

    while True:
        try:
            task = queue.get_nowait()
        except Exception:
            break
        person, env, view, first, last = task
        out_dir, frames = pending_frames(args, *task)
        if args.limit:
            frames = frames[: args.limit]
        tag = f"{person}/{env}/{view}"
        if not frames:
            print(f"[gpu{gpu}.{rank}] {tag}: 已完成", flush=True)
            continue
        wanted, t0, n_ok, n_none = set(frames), time.time(), 0, 0
        cap = cv2.VideoCapture(str(Path(args.video_root) / person / env / f"{view}.mp4"))
        idx = -1
        while idx < frames[-1]:
            idx += 1
            if not cap.grab():
                break
            if idx not in wanted:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    outputs = estimator.process_one_image(img=img, bboxes=None)
                best, _ = select_best_person(outputs, verbose=False)
            except Exception as e:  # 单帧出错不影响其余帧，重跑时会再试
                print(f"[gpu{gpu}.{rank}] {tag} 帧 {idx} 出错: {e!r}", flush=True)
                torch.cuda.synchronize()
                continue
            if best is None:
                n_none += 1
                out_dir.mkdir(parents=True, exist_ok=True)
                with (out_dir / NONE_FILE).open("a") as f:
                    f.write(f"{idx}\n")
                continue
            best["frame"], best["frame_idx"] = img, idx
            save_frame(best, out_dir, idx)
            n_ok += 1
        cap.release()
        dt = time.time() - t0
        print(
            f"[gpu{gpu}.{rank}] {tag}: 完成 {n_ok} 帧, 未检出 {n_none}, "
            f"{dt / max(1, n_ok + n_none):.2f} s/帧",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-root", default="/mnt/drive/videos_split")
    ap.add_argument("--result-root", default="/mnt/drive/sam3d_body_results_right_full")
    ap.add_argument("--marks", default="/mnt/drive/annotation/split_mid_end/mini.json")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--fp16-detector", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="每个任务最多处理的帧数（测试用）")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tasks = build_tasks(args)
    counts = {t: len(pending_frames(args, *t)[1]) for t in tasks}
    print(f"任务 {len(tasks)} 个, 待补 {sum(counts.values())} 帧", flush=True)
    if args.dry_run:
        return
    tasks = [t for t in tasks if counts[t]]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    for t in tasks:
        queue.put(t)
    time.sleep(1)
    procs = [
        ctx.Process(target=worker, args=(g, r, queue, args))
        for g in args.gpus
        for r in range(args.workers_per_gpu)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    left = sum(len(pending_frames(args, *t)[1]) for t in tasks)
    bad = [p.exitcode for p in procs if p.exitcode]
    print(f"结束: 剩余 {left} 帧, 异常退出进程 {len(bad)}", flush=True)
    raise SystemExit(1 if bad or left else 0)


if __name__ == "__main__":
    main()

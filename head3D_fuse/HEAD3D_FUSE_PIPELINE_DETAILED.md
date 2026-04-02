# head3D_fuse 详细流程文档

## 1. 文档目标

本文档用于说明 head3D_fuse 模块从多视角 3D 关键点输入，到融合、平滑、评估、可视化输出的完整执行流程。内容对应当前代码实现，便于二次开发、排错与实验复现。

适用入口：
- head3D_fuse/main.py
- head3D_fuse/infer.py
- configs/head3d_fuse.yaml

---

## 2. 总体架构

> 🔍 **关于融合策略的详细说明**，请参考 [FUSION_STRATEGY_DETAILED.md](FUSION_STRATEGY_DETAILED.md)。
> 本文档重点讲流程与调度，策略文档重点讲融合算法、对齐方式与参数选择。

模块采用两层架构：

1. 调度层（main.py）
- 读取 Hydra 配置
- 收集 Person/Env 任务
- 多进程分发任务
- 统一等待任务结束

2. 执行层（infer.py）
- 对单个 Person/Env 执行完整流水线：
  - 多视角帧对齐与读取
  - 逐帧融合
  - 时序平滑
  - 融合前后对比评估
  - 融合与单视角对比评估

---

## 3. 输入输出约定

### 3.1 输入数据

默认配置下的关键输入目录：

- sam3d 结果根目录：/workspace/data/sam3d_body_results_right_full
- 融合结果输出根目录：/workspace/data/head3d_fuse_results
- 视频目录：/workspace/data/videos_split
- 帧范围标注：/workspace/data/annotation/split_mid_end/mini.json

Person/Env 目录组织（示意）：

sam3d_results_root/
- 01/
  - 夜多い/
  - 夜少ない/
  - 昼多い/
  - 昼少ない/
- 02/
- ...

### 3.2 输出数据

每个 Person/Env 会产生两类根输出：

1. 推理结果输出（infer_root）
- fused_npz/: 融合后的逐帧 npy/npz
- smoothed_fused_npz/: 平滑后的逐帧 npy/npz

2. 日志与可视化输出（log_root）
- workers_logs/worker_x.log
- env_logs/{person}_{env}.log
- {person}/{env}/fused/
- {person}/{env}/smoothed/
- {person}/{env}/comparison/
- {person}/{env}/fused_vs_views_comparison/
- {person}/{env}/merged_video/

---

## 4. 调度层流程（main.py）

### 4.1 配置加载

入口函数使用 Hydra：
- config_path: ../configs
- config_name: head3d_fuse

读取的核心参数：
- cfg.paths.sam3d_results_path
- cfg.paths.result_output_path
- cfg.log_path
- cfg.infer.person_list
- cfg.infer.env_list
- cfg.infer.workers

### 4.2 任务收集

遍历 sam3d_results_path 下所有 Person 目录，再遍历各 Env 目录：

1. Person 过滤
- 若 person_list 包含 -1，表示不过滤（全量）
- 否则仅处理 person_list 中指定 ID

2. Env 过滤
- 若 env_list 包含 all，表示不过滤（全量）
- 否则仅处理 env_list 指定环境

收集后的任务单元是 env_dir（一个 Person/Env）。

### 4.3 多进程切分

- 使用 numpy.array_split 将任务按 workers 切分
- 每个子进程运行 _worker
- 启动方式固定为 spawn（mp.set_start_method("spawn", force=True)）

### 4.4 进程内日志

每个 worker 会初始化：

1. worker 汇总日志
- log_root/workers_logs/worker_{id}.log

2. 每个 Person/Env 独立日志
- out_root/env_logs/{person}_{env}.log

然后对分配到的 env_dir 顺序调用：
- process_single_person_env(env_dir, out_root, infer_root, cfg)

---

## 5. 执行层流程（infer.py）

单个 Person/Env 的标准执行顺序如下。

### 5.1 帧对齐与输入准备

函数：process_single_person_env

1. 获取 view_list（默认 front/left/right）
2. 读取 start-mid-end 标注（用于有效帧范围）
3. 调用 assemble_view_npz_paths 对齐三视角帧
4. 若对齐后无可用帧，则直接返回

### 5.2 多视角融合（_fuse_single_person_env）

逐帧处理 frame_triplets：

1. 读取各视角 npz 输出
2. 归一化关键点（处理 batch 维、关键点缺失、索引截取）
3. 检查整视角是否全 NaN
- 任一视角全 NaN 时，该帧跳过
4. 调用 fuse_3view_keypoints 执行融合
- 支持方法：median/mean 等
- 支持坐标变换与对齐配置：
  - transform_mode
  - view_transforms
  - alignment_method
  - alignment_reference
  - alignment_scale
  - alignment_trim_ratio
  - alignment_max_iters
5. 保存融合结果到 fused_npz
6. 生成可视化帧
- 各视角单独可视化
- 融合对比可视化

循环结束后：
- 输出 npz 差异报告（如果有）
- 合并帧为视频（fused_3d_keypoints + front/left/right）

### 5.3 时序平滑（_smooth_fused_keypoints_env）

触发条件：
- cfg.smooth.enable_temporal_smooth 为 true
- all_fused_kpts 非空

处理步骤：

1. 将 dict[frame_idx -> kpt] 堆叠为 (T, N, 3)
2. 根据 temporal_smooth_method 组装参数：
- gaussian: sigma
- savgol: window_length, polyorder
- kalman: process_variance, measurement_variance
- bilateral: sigma_space, sigma_range
3. 调用 smooth_keypoints_sequence
4. 保存平滑结果到 smoothed_fused_npz
5. 生成 smoothed 融合可视化帧并合成视频

### 5.4 平滑前后评估（_compare_fused_smoothed_keypoints）

触发条件：
- keypoints_array 与 smoothed_array 均非空
- cfg.smooth.enable_comparison 为 true

输出目录：
- {log_root}/{person}/{env}/comparison/

主要输出：
- smoothing_metrics.json
- smoothing_comparison_report.txt
- trajectory_comparison.png（可选）
- metrics_comparison.png（可选）

关键指标包括：
- mean_difference
- jitter_reduction
- acceleration_reduction

### 5.5 融合与单视角评估（_compare_fused_with_views）

触发条件：
- cfg.fuse.enable_fused_view_comparison 为 true
- all_fused_kpts 非空

输出目录：
- {log_root}/{person}/{env}/fused_vs_views_comparison/

主要输出：
- fused_vs_views_metrics.json
- fused_vs_views_report.txt
- fused_vs_views_comparison.png（可选）

核心指标包括：
- 融合结果到各视角平均距离
- 融合结果到视角质心距离
- 融合结果时序抖动
- 视角间一致性

---

## 6. 配置项速查（head3d_fuse.yaml）

### 6.1 infer

- workers: 并行进程数
- person_list: 处理的人 ID；[-1] 表示全部
- env_list: 处理的环境；[all] 表示全部
- view_list: 视角列表，默认 [front, left, right]

### 6.2 fuse

- fuse_method: 融合方法（常用 median）
- transform_mode/view_transforms: 相机坐标变换
- alignment_method: none/procrustes/procrustes_trimmed
- enable_fused_view_comparison: 是否启用融合与单视角评估
- enable_fused_view_comparison_plots: 是否生成评估图

### 6.3 smooth

- enable_temporal_smooth: 是否启用时序平滑
- temporal_smooth_method: gaussian/savgol/kalman/bilateral
- 对应方法参数：sigma/window_length/polyorder 等
- enable_comparison: 是否启用平滑前后评估
- enable_comparison_plots: 是否输出评估图

---

## 7. 运行方式

在 code 目录执行：

python -m head3D_fuse.main

常见覆盖参数（Hydra 形式）示例：

python -m head3D_fuse.main infer.workers=8 infer.person_list=[1,2,3] infer.env_list=[夜多い,昼多い]

仅关闭平滑：

python -m head3D_fuse.main smooth.enable_temporal_smooth=false

开启并限制融合对比关键点：

python -m head3D_fuse.main fuse.enable_fused_view_comparison=true fuse.fused_view_comparison_keypoint_indices=[0,1,2,3,4,5,6]

---

## 8. 日志与排错建议

1. 先看 worker 汇总日志
- workers_logs/worker_{id}.log

2. 再看具体任务日志
- env_logs/{person}_{env}.log

3. 常见问题定位
- No aligned frames found: 三视角帧对齐后为空，检查输入缺帧或标注范围
- Missing pred_keypoints_3d: 某视角该帧输出缺失
- all NaN skipped: 某视角整帧无效导致当前帧跳过
- Not enough view data for comparison: 视角数据不足以做对比

4. 性能建议
- workers 不宜盲目调大，建议按 CPU 核数与 IO 压力逐步测试
- 若主要瓶颈在可视化与视频合成，可先关闭部分图像输出做定位

---

## 9. 实际落地建议

1. 数据质量优先
- 在融合前增加对各视角缺失率统计，可提前识别坏段视频

2. 评估闭环
- 每次改动融合或平滑参数后，固定一个 person/env 子集做 A/B 指标对比

3. 生产化输出规范
- 建议把关键配置快照（完整 YAML）与结果一起落盘，保证可复现

---

## 10. 流程总览（简版）

1. main.py 读取配置并收集 Person/Env 任务
2. 多进程 worker 并行处理每个 env_dir
3. infer.py 对单任务执行：
- 帧对齐
- 融合并保存 fused_npz
- 时序平滑并保存 smoothed_fused_npz
- 平滑前后评估
- 融合与单视角评估
4. 输出日志、图表、报告、视频

这就是当前 head3D_fuse 的完整执行链路。

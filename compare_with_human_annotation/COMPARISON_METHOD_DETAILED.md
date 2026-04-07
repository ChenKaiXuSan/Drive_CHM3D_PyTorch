# 头部姿态与人工标注比较方法详细说明

## 1. 文档目的

本文档系统说明 compare_with_human_annotation 模块中的“自动估计结果 vs 人工标注”比较方法，包括：

- 输入数据与标注解析方式
- 头部角度计算方法（Pitch / Yaw / Roll）
- 标注方向到角度方向的映射规则
- 帧级匹配判定逻辑
- 序列级、人物级、环境级统计方法
- 输出文件格式与关键字段
- 方法假设与局限

适用代码实现位于：

- main.py
- angle_calculator.py
- load.py
- batch_run.py

## 2. 任务定义

比较任务可形式化为：

给定某视频序列中每一帧的 3D 关键点估计结果，计算头部角度；再与该帧对应的人工时间段标注进行方向一致性判断，得到匹配与不匹配统计。

最小比较单元是“帧-标注对”（frame-label pair），不是“帧”本身。

这意味着：

- 一帧可对应 0 个标注（不参与比较）
- 一帧可对应 1 个标注
- 一帧也可对应多个重叠标注（每个标注都单独比较并计数）

## 3. 输入数据

## 3.1 关键点输入

每帧对应一个 npy 文件，命名格式：

- frame_{frame_idx:06d}_fused.npy

文件中读取字段：

- fused_keypoints_3d

期望形状：

- (70, 3)

若存在 batch 维（即 3 维数组），实现会取第 0 个样本后再截取前 70 个关键点。

## 3.2 标注输入

标注由 JSON 加载，解析为结构体 HeadMovementLabel：

- start_frame: int
- end_frame: int
- label: str

当前数据中，每个视频条目包含 3 位标注者的 `annotations`；实现会将这 3 份标注全部合并到同一个视频的标注列表中，再进入后续比较流程。

每个标签表示一个时间区间，某帧 frame_idx 满足：

- start_frame <= frame_idx <= end_frame

则判定该帧属于该标签。

## 4. 关键点选择与有效性检查

参与角度计算的关键点：

- nose
- left_eye
- right_eye
- left_ear
- right_ear
- left_shoulder
- right_shoulder

有效性规则：

- 若任一所需关键点含非有限值（NaN/Inf），该帧直接跳过（该帧不产出角度，不参与后续比较）。

## 5. 角度计算方法

核心在 angle_calculator.py 的 calculate_head_angles。

## 5.1 中间点定义

- eye_center = (left_eye + right_eye) / 2
- ear_center = (left_ear + right_ear) / 2
- shoulder_center = (left_shoulder + right_shoulder) / 2

说明：当前 ear_center 已计算但未直接用于最终角度公式。

## 5.2 Pitch（俯仰角）

定义向量：

- v_ns = nose - shoulder_center

将其在水平面上的投影长度记为：

$$
 d_h = \sqrt{v_{ns,x}^2 + v_{ns,z}^2}
$$

Pitch 角：

$$
\text{pitch} = \arctan2(v_{ns,y}, d_h)
$$

并转换为角度制（degree）。

符号约定：

- pitch > 0: 抬头
- pitch < 0: 低头

## 5.3 Yaw（偏航角）

先构造头部朝向参考向量：

- face_forward = nose - eye_center

再取其水平分量：

- face_forward_horizontal = [face_forward_x, 0, face_forward_z]

归一化后，与相机前向 -Z 轴比较，采用：

$$
\text{yaw} = \arctan2(f_x, -f_z)
$$

其中 $f_x, f_z$ 为归一化后水平向量分量。

退化处理：

- 若水平向量范数过小（<= 1e-6），yaw 置为 0.0。

符号约定：

- yaw > 0: 向右转
- yaw < 0: 向左转

## 5.4 Roll（翻滚角）

用双眼连线向量：

- eye_vec = right_eye - left_eye

在 X-Y 平面计算：

$$
\text{roll} = \arctan2(eye\_vec_y, eye\_vec_x)
$$

说明：当前比较逻辑仅使用 pitch 和 yaw，roll 暂不参与匹配判定。

## 6. 标签系统与方向映射

当前支持 5 类基本方向：

- front
- up
- down
- left
- right

映射为二维方向符号 (expected_pitch_dir, expected_yaw_dir)：

- front -> (0, 0)
- up -> (1, 0)
- down -> (-1, 0)
- left -> (0, -1)
- right -> (0, 1)

语义解释：

- 1: 正方向
- -1: 负方向
- 0: 中立（绝对值在阈值内）

## 7. 帧级匹配判定

给定阈值 $\tau$（单位：度，默认 15），单轴判定函数 direction_match 规则为：

- expected_dir = 1 时：angle_value > tau
- expected_dir = -1 时：angle_value < -tau
- expected_dir = 0 时：|angle_value| <= tau

对某个标注标签的最终匹配条件：

- pitch 轴匹配 AND yaw 轴匹配

即：

$$
\text{is_match} = \text{match_pitch} \land \text{match_yaw}
$$

这意味着：

- 任一轴不满足阈值方向条件，即判不匹配
- front 要求 pitch 与 yaw 都在中立范围内

## 8. 序列比较流程

单序列处理流程如下：

1. 遍历 fused_dir 中全部 frame_*_fused.npy
2. 对每帧计算 pitch/yaw/roll（失败帧跳过）
3. 对有角度结果的帧，查询该帧所有标注
4. 对每个标注执行方向匹配，产生 frame-label 级比较结果
5. 汇总为：
   - angles: 每帧角度
   - comparisons: 仅包含有标注且完成比较的帧

## 9. 统计口径

## 9.1 单 person-env 统计

batch_run.py 中 run_single_comparison 输出：

- total_frames: 成功产出角度的帧数
- annotated_frames: comparisons 中帧数（至少有一个可比较标签）
- total_annotations: 全部 frame-label 对数量
- total_matches: is_match=True 的 frame-label 对数量
- match_rate: total_matches / total_annotations * 100
- by_direction: 按标签方向分组统计 total/matched/rate

注意：match_rate 基于“标注数”而不是“帧数”。

## 9.2 全量批处理统计

run_batch_comparison 对所有 person-env 组合重复上述流程，额外输出：

- 每个组合的 result.json
- 全部组合汇总 summary.json
- 论文风格文本报告 paper_report.txt

报告包含：

- 总体统计
- 按环境统计
- 按方向统计
- 逐组合明细表

## 9.3 多数投票批处理统计

run_batch_comparison_majority_vote 会先对同一视频下的 3 位标注者做帧级多数投票，再将投票后的标注与模型结果比较。该模式的输出与标准批处理相互独立，会写入：

- /workspace/data/compare_with_human_annotation_results_majority_vote

目录结构与标准批处理一致，但文件内容来自多数投票后的标注结果。

## 10. 输出结构

默认输出目录：

- /workspace/data/comparison_results

层级：

- {person_id}/{env_en}/result.json
- paper_report.txt
- summary.json

其中 env_en 为：

- day_high / day_low / night_high / night_low

## 11. 阈值参数说明

阈值参数 threshold_deg 在两种模式中可配置：

- single: 对单组合比较生效
- all/batch: 对全量比较与报告生效

阈值越大，方向判定更严格。

对 expected_dir = 0（如 front 轴）来说，阈值越大，中立区间越宽；
对 expected_dir = ±1（如 left/right/up/down 对应轴）来说，阈值越大，判定正负方向越严格。

## 12. 方法假设与局限

## 12.1 关键假设

- 相机坐标系满足实现中的符号约定（例如 -Z 为前向）
- 标注标签语义与 pitch/yaw 的正负方向一致
- 70 点关键点中头肩关键点质量稳定

## 12.2 局限

- 当前匹配不使用 roll，无法评价左右倾斜动作与标注一致性
- yaw 计算依赖 nose-eye_center 几何关系，受关键点抖动影响较大
- 若标注存在重叠，同一帧会产生多个比较项，可能放大某些区间权重
- front 标签使用双轴同时中立判定，较严格
- 阈值为全局固定值，未按人物/环境自适应

## 13. 建议的扩展方向

- 引入 roll 相关标签并纳入匹配
- 增加帧级平滑（时域滤波）后再比较，提高稳健性
- 报告中补充混淆矩阵（标签方向 vs 估计方向）
- 基于验证集选择分环境阈值，替代单一全局阈值
- 加入对无效关键点比例的质量统计

## 14. 关键函数索引

- load.py
  - load_fused_keypoints
  - load_head_movement_annotations
  - get_all_annotations_for_frame
- angle_calculator.py
  - extract_head_keypoints
  - calculate_head_angles
  - direction_match
  - classify_label
- main.py
  - HeadPoseAnalyzer.analyze_head_pose
  - HeadPoseAnalyzer.compare_with_annotations
  - HeadPoseAnalyzer.analyze_sequence_with_annotations
- batch_run.py
  - run_single_comparison
  - run_batch_comparison
  - generate_paper_report

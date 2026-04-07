# 与人工标注比较模块 (Comparison with Human Annotation)

## 功能说明

该模块用于从融合后的3D关键点数据中计算头部的三个转动角度，并与人工标注进行比较，评估预测的准确性。

### 支持的角度指标

1. **Pitch（俯仰角）**：上下点头的角度
   - 正值：抬头
   - 负值：低头
   - 范围：约 -90° 到 +90°

2. **Yaw（偏航角）**：左右转头的角度
   - 正值：向右转
   - 负值：向左转
   - 范围：约 -180° 到 +180°

3. **Roll（翻滚角）**：头部左右倾斜的角度
   - 正值：向右倾斜
   - 负值：向左倾斜
   - 范围：约 -180° 到 +180°

## 快速开始

## 方法文档

比较方法的完整说明见：

- [COMPARISON_METHOD_DETAILED.md](COMPARISON_METHOD_DETAILED.md)

### 命令行运行

默认直接运行全部人物和全部环境：

```bash
python -m compare_with_human_annotation
```

先对 3 位标注者做多数投票，再运行全量比较：

```bash
python -m compare_with_human_annotation majority
```

也可以显式指定全量模式和阈值：

```bash
python -m compare_with_human_annotation all --threshold 15
```

运行单个人物与环境：

```bash
python -m compare_with_human_annotation single 01 昼多い --threshold 15
```

运行单视角 SAM3D 结果与标注比较（支持 front/left/right/all）：

```bash
python -m compare_with_human_annotation single_view 01 昼多い --view all --annotation-mode majority --threshold 5
python -m compare_with_human_annotation single_view 01 昼多い --view all --annotation-mode by_annotator --threshold 5
```

单视角模式输出路径：

```text
/workspace/data/compare_with_human_annotation_results/sam3d_views/majority/
/workspace/data/compare_with_human_annotation_results/sam3d_views/by_annotator/annotator_1/
/workspace/data/compare_with_human_annotation_results/sam3d_views/by_annotator/annotator_2/
/workspace/data/compare_with_human_annotation_results/sam3d_views/by_annotator/annotator_3/
```

多数投票模式的输出会单独写到：

```text
/workspace/data/compare_with_human_annotation_results_majority_vote/
```

### 1. 单帧分析

```python
from pathlib import Path
from compare_with_human_annotation import HeadPoseAnalyzer

# 创建分析器（不进行标注比较）
analyzer = HeadPoseAnalyzer()

# 分析单个帧
npy_path = Path("/path/to/frame_000619_fused.npy")
result = analyzer.analyze_head_pose(npy_path)

if result:
    print(f"俯仰角 (Pitch): {result['pitch']:.2f}°")
    print(f"偏航角 (Yaw): {result['yaw']:.2f}°")
    print(f"翻滚角 (Roll): {result['roll']:.2f}°")
```

### 2. 序列分析（不涉及标注比较）

```python
from pathlib import Path
from compare_with_human_annotation import HeadPoseAnalyzer

analyzer = HeadPoseAnalyzer()
fused_dir = Path("/path/to/fused_npz")
results = analyzer.analyze_sequence(fused_dir, start_frame=0, end_frame=1000)

for frame_idx in sorted(results.keys()):
    angles = results[frame_idx]
    print(f"Frame {frame_idx}: "
          f"Pitch={angles['pitch']:6.2f}°, "
          f"Yaw={angles['yaw']:6.2f}°, "
          f"Roll={angles['roll']:6.2f}°")
```

### 3. 与标注比较（核心功能）

```python
from pathlib import Path
from compare_with_human_annotation import (
    HeadPoseAnalyzer,
    load_head_movement_annotations,
)

# 加载标注
annotations = load_head_movement_annotations(
    Path("/path/to/annotations.json")
)

# 创建分析器
analyzer = HeadPoseAnalyzer(annotation_dict=annotations)

# 分析序列并与标注比较
results = analyzer.analyze_sequence_with_annotations(
    video_id="01_day_high",
    fused_dir=Path("/path/to/fused_npz"),
    start_frame=0,
    end_frame=1000,
)

# 查看角度结果
angles = results["angles"]
print(f"分析了 {len(angles)} 帧")

# 查看与标注的比较结果
comparisons = results["comparisons"]
for frame_idx, comparison in sorted(comparisons.items()):
    print(f"\nFrame {frame_idx}:")
    print(f"  计算角度: Pitch={comparison['angles']['pitch']:.2f}°, "
          f"Yaw={comparison['angles']['yaw']:.2f}°")
    
    # 显示匹配结果
    for match in comparison['matches']:
        annotation = match['annotation']
        is_match = match['is_match']
        status = "✓ 匹配" if is_match else "✗ 不匹配"
        print(f"  标注: {annotation.label} ({annotation.start_frame}-{annotation.end_frame}) {status}")
```

### 4. 使用不同阈值进行比较

```python
analyzer = HeadPoseAnalyzer(annotation_dict=annotations)

# 使用不同的阈值
for threshold in [5, 10, 15, 20, 25, 30]:
    results = analyzer.analyze_sequence_with_annotations(
        video_id="01_day_high",
        fused_dir=fused_dir,
    )
    
    # 计算匹配率
    total_matches = 0
    successful_matches = 0
    
    for frame_idx, comparison in results["comparisons"].items():
        for match_info in comparison["matches"]:
            if match_info["is_match"]:
                successful_matches += 1
            total_matches += 1
    
    if total_matches > 0:
        match_rate = (successful_matches / total_matches) * 100
        print(f"阈值 {threshold:2d}°: {successful_matches:3d}/{total_matches:3d} = {match_rate:5.1f}%")
```

## 数据格式

### 输入数据

- **融合关键点文件格式**：`.npy` 文件
- **文件名格式**：`frame_{frame_idx:06d}_fused.npy`
- **数据结构**：
  ```python
  {
      'fused_keypoints_3d': np.ndarray  # 形状 (70, 3)
  }
  ```

### 标注JSON格式

```json
[
    {
        "data": {
            "video": "person_01_day_high_1.mp4"
        },
        "annotations": [
            {
                "result": [
                    {
                        "type": "timelinelabels",
                        "value": {
                            "ranges": [
                                {"start": 100, "end": 200},
                                {"start": 300, "end": 400}
                            ],
                            "timelinelabels": ["left", "right"]
                        }
                    }
                ]
            }
        ]
    }
]
```

## 标注标签定义

支持的标注标签及其对应的角度方向：

| 标签 | 含义 | Pitch | Yaw |
|------|------|-------|-----|
| `front` | 朝向正前方 | 0 | 0 |
| `up` | 向上点头 | +1 | 0 |
| `down` | 向下点头 | -1 | 0 |
| `left` | 向左转头 | 0 | -1 |
| `right` | 向右转头 | 0 | +1 |

## 核心API

### `HeadPoseAnalyzer` 类

```python
class HeadPoseAnalyzer:
    def __init__(self, annotation_dict=None):
        # annotation_dict: 可选的标注字典 {video_id: [HeadMovementLabel, ...]}
        
    def analyze_head_pose(npy_path) -> Dict[str, float]:
        # 分析单帧
        
    def analyze_sequence(fused_dir, start_frame=None, end_frame=None) -> Dict[int, Dict]:
        # 分析序列
        
    def compare_with_annotations(video_id, frame_idx, angles, threshold_deg=15.0) -> Dict:
        # 与标注比较单帧
        
    def analyze_sequence_with_annotations(video_id, fused_dir, ...) -> Dict:
        # 分析序列并与标注比较（核心方法）
```

### 辅助函数

- `load_head_movement_annotations(json_path)` - 加载标注
- `load_multi_annotator_annotations(json_path)` - 加载多标注者标注
- `get_annotation_for_frame(annotations, frame_idx)` - 获取单个标注
- `get_all_annotations_for_frame(annotations, frame_idx)` - 获取所有标注

## 依赖

- numpy
- pathlib (标准库)
- logging (标准库)

## 作者

Kaixu Chen (chenkaixusan@gmail.com)

## 日期

February 7, 2026

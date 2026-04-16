# Drive-CHM3D: 3D Modeling of Compensatory Head Movements in Simulated Glaucomatous Driving

Drive-CHM3D is a multi-view computer vision framework for reconstructing and analyzing compensatory head movements in simulated glaucomatous driving scenarios.
The goal is to quantify how visual field restriction changes head-scanning behavior under different driving conditions.

## Why This Project

Glaucoma-induced visual field loss reduces peripheral awareness and makes hazard perception harder.
Drivers often compensate with additional head scanning, but reliable quantification is difficult because of viewpoint changes, occlusions, and illumination differences.

Drive-CHM3D addresses this challenge with a robust 3D analysis pipeline that combines synchronized multi-view observations into a unified and stable head-motion representation.

## Highlights

- Multi-view head-motion reconstruction from synchronized left/front/right cameras
- Cross-view alignment and coordinate-wise median fusion for robust 3D aggregation
- Temporal smoothing for physically plausible trajectories and reduced jitter
- Axis-wise head-pose evaluation (yaw and pitch)
- Human-annotation-based validation with tolerance-based directional comparison
- Behavioral analysis across visual field restriction, traffic density, and visibility settings

## Experimental Setting

### Driving Conditions

- Visual field restriction: none / medium / heavy
- Traffic density: low / heavy
- Visibility: daytime / nighttime

### Camera Setup

- Left, front, and right synchronized in-cabin views
- 30 FPS
- 720p resolution

### Participants

- 15 healthy volunteers
- Valid Japanese driver's licenses
- Normal or corrected-to-normal vision
- Simulated visual field loss using pinhole glasses

## Method Overview

### 1) Single-View 3D Keypoint Prediction

For each frame and each view:

- detect the driver,
- identify the target subject using spatial and temporal consistency,
- estimate sparse 3D keypoints with SAM-3D-Body.

Keypoints used for head/upper-body analysis include:

- nose
- left/right eyes
- left/right ears
- left/right shoulders
- left/right hands
- left/right acromion
- neck

### 2) Multi-View Fusion

To build a unified 3D representation:

- preprocess keypoints,
- remove invalid observations,
- filter unreliable frames,
- align left/right views to the front-view coordinate system (Procrustes),
- aggregate aligned keypoints using coordinate-wise median fusion.

This improves robustness against occlusion, viewpoint bias, and noisy single-view estimation.

### 3) Temporal Smoothing

Fused trajectories are refined with Gaussian smoothing to reduce frame-level jitter while preserving meaningful motion dynamics.

## Evaluation Protocol

### Fusion Quality

- distance to each single-view estimate
- distance to centroid
- inter-view consistency
- temporal jitter

### Temporal Stability

- mean trajectory difference
- velocity reduction
- acceleration reduction
- jitter reduction

### Head-Pose Accuracy

- yaw: left-right motion
- pitch: up-down motion

Predicted directional changes are compared against human annotations using tolerance-based decision rules.

## Main Findings

- Multi-view fusion outperforms single-view baselines.
- Fusion reduces viewpoint-dependent bias.
- Temporal smoothing improves motion stability.
- Under simulated visual field restriction, drivers show:
  - stronger horizontal scanning,
  - more frequent head movements,
  - stronger compensatory behavior in demanding settings (for example, low visibility and heavy traffic).

These results indicate that compensatory head movement is inherently multi-directional and cannot be captured reliably from a single viewpoint.

## Quick Start

### 1) Environment Setup

```bash
cd /workspace/code

# Recommended: use a dedicated conda env (example)
conda create -n mesh_3d python=3.10 -y
conda activate mesh_3d

pip install -r requirements.txt
pip install -e .
```

### 2) Run Multi-View Fusion

```bash
cd /workspace/code
python -m head3D_fuse.main
```

Run alignment comparison experiments:

```bash
cd /workspace/code
bash bash/run_alignment_compare_3exp.sh
```

### 3) Compare With Human Annotations

Single run:

```bash
cd /workspace/code
python -m compare_with_human_annotation.main run.mode=fused run.annotation_mode=majority run.threshold=10
```

Batch run for multiple modes/thresholds:

```bash
cd /workspace/code
bash bash/run_all_compare_with_human_annotation.sh
```

## Data and Output Locations

In this workspace, raw and processed assets are typically stored under `/workspace/data`, for example:

- `/workspace/data/multi_view_driver_action`
- `/workspace/data/head3d_fuse_results`
- `/workspace/data/head_pose_analysis_results`
- `/workspace/data/compare_with_human_annotation_results`

Project logs are typically written under:

- `/workspace/code/logs`

## Repository Structure (Current)

```bash
code/
├── analysis/                         # analysis notebooks and scripts
├── bash/                             # batch execution scripts
├── camera_calibration/               # camera calibration utilities
├── compare_with_human_annotation/    # annotation-based evaluation pipeline
├── configs/                          # configuration files
├── head3D_fuse/                      # multi-view fusion pipeline
├── head_movement_analysis/           # motion analysis modules
├── mesh/                             # mesh-related utilities
├── pegasus/                          # model / runtime components
├── SAM3Dbody/                        # SAM-3D-Body integration
├── tests/                            # tests
├── vis_3d_kpt/                       # 3D visualization utilities
├── requirements.txt
├── setup.py
└── README.md
```
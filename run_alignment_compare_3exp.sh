#!/usr/bin/env bash
set -euo pipefail

# Four alignment comparison experiments for head3D_fuse:
#   1) none
#   2) procrustes
#   3) procrustes_trimmed
#   4) extrinsics (view_transforms with calibrated R_wc/t_wc)
#
# Usage:
#   bash /workspace/code/run_alignment_compare_3exp.sh
#
# Optional environment overrides:
#   PERSONS='[1,2,3]' ENVS='[夜多い,昼多い]' WORKERS=4 \
#   bash /workspace/code/run_alignment_compare_3exp.sh
#   MAX_PARALLEL=4 bash /workspace/code/run_alignment_compare_3exp.sh

cd /workspace/code

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BASE_OUT="${BASE_OUT:-/workspace/data/head3d_fuse_results_align_cmp/${RUN_ID}}"
BASE_LOG="${BASE_LOG:-/workspace/code/logs/head3d_fuse_align_cmp/${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/sam_3d_body/bin/python}"

# Customize these as needed
PERSONS="${PERSONS:-[1]}"
ENVS="${ENVS:-[all]}"
WORKERS="${WORKERS:-8}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# Common options
COMMON_ARGS=(
  "infer.person_list=${PERSONS}"
  "infer.env_list=${ENVS}"
  "infer.workers=${WORKERS}"
  "fuse.fuse_method=median"
)

run_exp() {
  local method="$1"
  shift

  echo "============================================================"
  echo "Running method: ${method}"
  echo "Output dir: ${BASE_OUT}/${method}"
  echo "Log dir:    ${BASE_LOG}/${method}"
  echo "Python:     ${PYTHON_BIN}"
  echo "============================================================"

  "${PYTHON_BIN}" -m head3D_fuse.main \
    "${COMMON_ARGS[@]}" \
    "fuse.alignment_method=${method}" \
    "$@" \
    "paths.result_output_path=${BASE_OUT}/${method}" \
    "log_path=${BASE_LOG}/${method}"
}

wait_for_slot() {
  while true; do
    local running
    running=$(jobs -pr | wc -l)
    if [[ "${running}" -lt "${MAX_PARALLEL}" ]]; then
      break
    fi
    sleep 1
  done
}

PIDS=()
METHODS=()

launch_exp() {
  local method="$1"
  shift

  wait_for_slot

  run_exp "${method}" "$@" &
  local pid=$!
  PIDS+=("${pid}")
  METHODS+=("${method}")
  echo "Launched ${method} (pid=${pid})"
}

# 1) No alignment
launch_exp "none"

# 2) Standard Procrustes
launch_exp "procrustes" \
  "fuse.alignment_reference=front" \
  "fuse.alignment_scale=true"

# 3) Robust Procrustes (trimmed)
launch_exp "procrustes_trimmed" \
  "fuse.alignment_reference=front" \
  "fuse.alignment_scale=true" \
  "fuse.alignment_trim_ratio=0.2" \
  "fuse.alignment_max_iters=3"

# 4) Extrinsics-based alignment (world_to_camera)
#    Uses R_wc / t_wc found from mesh_triangulation camera layout:
#      front: R=[[-1,0,0],[0,0,-1],[0,-1,0]],             t=[0,1.5,0.62]
#      left:  R=[[-0.91129,-0.411765,0],[0,0,-1],[0.411765,-0.91129,0]], t=[0,1.5,0.85]
#      right: R=[[-0.91129, 0.411765,0],[0,0,-1],[-0.411765,-0.91129,0]], t=[0,1.5,0.85]
launch_exp "extrinsics" \
  "fuse.alignment_method=none" \
  "fuse.transform_mode=world_to_camera" \
  "fuse.view_transforms.front.R=[[-1.0,0.0,0.0],[0.0,0.0,-1.0],[0.0,-1.0,0.0]]" \
  "fuse.view_transforms.front.t_wc=[0.0,1.5,0.62]" \
  "fuse.view_transforms.left.R=[[-0.91129,-0.411765,0.0],[0.0,0.0,-1.0],[0.411765,-0.91129,0.0]]" \
  "fuse.view_transforms.left.t_wc=[0.0,1.5,0.85]" \
  "fuse.view_transforms.right.R=[[-0.91129,0.411765,0.0],[0.0,0.0,-1.0],[-0.411765,-0.91129,0.0]]" \
  "fuse.view_transforms.right.t_wc=[0.0,1.5,0.85]"

echo ""
echo "Waiting for ${#PIDS[@]} experiment(s) to finish..."

FAIL_COUNT=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  method="${METHODS[$i]}"
  if wait "${pid}"; then
    echo "[OK] ${method} (pid=${pid})"
  else
    echo "[FAIL] ${method} (pid=${pid})"
    FAIL_COUNT=$((FAIL_COUNT + 1))
  fi
done

echo ""
if [[ "${FAIL_COUNT}" -gt 0 ]]; then
  echo "All experiments finished with ${FAIL_COUNT} failure(s)."
  exit 1
fi

echo "All experiments finished successfully."
echo "Results root: ${BASE_OUT}"
echo "Logs root:    ${BASE_LOG}"

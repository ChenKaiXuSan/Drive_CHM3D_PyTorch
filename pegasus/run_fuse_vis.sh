#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N head3d_fuse_vis
#PBS -t 0-21
#PBS -o logs/pegasus/head3d_fuse_vis_group_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/head3d_fuse_vis_group_${PBS_SUBREQNO}_err.log

# === 1. 環境準備 ===
cd /work/SSR/share/code/Drive_Face_Mesh_PyTorch

mkdir -p logs/pegasus/
# Activate conda environment
source "${CONDA_PREFIX}/etc/profile.d/conda.sh"
conda activate /home/SSR/luoxi/miniconda3/envs/multiview-video-cls

conda env list

# 一个数组任务对应一个人（01~21,24）
# 如需改人数，请同步修改上面的 #PBS -t 范围
PERSON_IDS=(
    01 02 03 04 05 06 07 08 09 10 11 12
    13 14 15 16 17 18 19 20 21 24
)

if [ -z "${PBS_SUBREQNO}" ]; then
    echo "[ERROR] PBS_SUBREQNO 未设置，请通过 qsub 的数组任务提交"
    exit 1
fi

if [ "${PBS_SUBREQNO}" -lt 0 ] || [ "${PBS_SUBREQNO}" -ge "${#PERSON_IDS[@]}" ]; then
    echo "[ERROR] PBS_SUBREQNO=${PBS_SUBREQNO} 超出范围 (0-$(( ${#PERSON_IDS[@]} - 1 )))"
    exit 1
fi

PERSON_ID=${PERSON_IDS[$PBS_SUBREQNO]}

echo "Node Index: ${PBS_SUBREQNO}"
echo "Processing person: ${PERSON_ID}"

# === 3. 可视化路径与运行参数 ===
HEAD3D_DIR="/work/SSR/share/data/drive/head3d_fuse_results"
VIDEO_PATH="/work/SSR/share/data/drive/videos_split"
SAM3D_RESULTS_PATH="/work/SSR/share/data/drive/sam3d_body_results_right_full"
VIS_OUT_PATH="/work/SSR/share/data/drive/fused_3d_vis"

# vis_3d_kpt 内部日志（all/error）保存目录
VIS_LOG_DIR="/work/SSR/share/code/Drive_Face_Mesh_PyTorch/logs/vis_3d_kpt/person_${PERSON_ID}"

# person 内部 env 并发数（建议 1-2，内存更稳）
NUM_WORKERS=1

echo "🏁 Node ${PBS_SUBREQNO} started at: $(date)"
echo "Head3D Dir: ${HEAD3D_DIR}"
echo "Video Path: $VIDEO_PATH"
echo "SAM3D Results Path: $SAM3D_RESULTS_PATH"
echo "Vis Output Path: ${VIS_OUT_PATH}"
echo "Vis Log Dir: ${VIS_LOG_DIR}"
echo "Num Workers: ${NUM_WORKERS}"

python -m vis_3d_kpt.main \
    --head3d-dir "${HEAD3D_DIR}" \
    --video-dir "${VIDEO_PATH}" \
    --sam3d-results-dir "${SAM3D_RESULTS_PATH}" \
    --out-dir "${VIS_OUT_PATH}" \
    --person-list "${PERSON_ID}" \
    --log-dir "${VIS_LOG_DIR}"

echo "🏁 Node ${PBS_SUBREQNO} finished at: $(date)"
# 一个 node 跑一个 person，该 person 下多个 env 由 --num-workers 控制并发
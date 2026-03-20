#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N head3d_fuse
#PBS -t 0-17
#PBS -o logs/pegasus/head3d_fuse_group_${PBS_SUBREQNO}.log
#PBS -e logs/pegasus/head3d_fuse_group_${PBS_SUBREQNO}_err.log

# === 1. 環境準備 ===
cd /work/SSR/share/code/Drive_Face_Mesh_PyTorch

mkdir -p logs/pegasus/

module load intelpython/2022.3.1
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
conda deactivate
# conda activate /home/SSR/luoxi/miniconda3/envs/sam_3d_body
conda activate /home/SSR/luoxi/miniconda3/envs/multiview-video-cls

conda env list

# 一个数组任务对应一个人（01~21,24）
# PERSON_IDS=(01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18) 
PERSON_IDS=(19 20 21 24)

if [ -z "${PBS_SUBREQNO}" ]; then
    echo "[ERROR] PBS_SUBREQNO 未设置，请通过 qsub 的数组任务提交"
    exit 1
fi

if [ "${PBS_SUBREQNO}" -lt 0 ] || [ "${PBS_SUBREQNO}" -ge "${#PERSON_IDS[@]}" ]; then
    echo "[ERROR] PBS_SUBREQNO=${PBS_SUBREQNO} 超出范围 (0-$(( ${#PERSON_IDS[@]} - 1 )))"
    exit 1
fi

PERSON_ID=${PERSON_IDS[$PBS_SUBREQNO]}
PERSON_LIST="[${PERSON_ID}]"

echo "Node Index: ${PBS_SUBREQNO}"
echo "Processing person: ${PERSON_ID}"

# === 3. パス設定と実行 ===
VIDEO_PATH="/work/SSR/share/data/drive/videos_split"
SAM3D_RESULTS_PATH="/work/SSR/share/data/drive/sam3d_body_results_right"
HEAD3D_OUT_PATH="/work/SSR/share/data/drive/head3d_fuse_results"
START_MID_END_PATH="/work/SSR/share/data/drive/annotation/split_mid_end/mini.json"

echo "🏁 Node ${PBS_SUBREQNO} started at: $(date)"
echo "Video Path: $VIDEO_PATH"
echo "SAM3D Results Path: $SAM3D_RESULTS_PATH"
echo "Head3D Output Path: $HEAD3D_OUT_PATH"

python -m head3D_fuse.main \
    paths.video_path=${VIDEO_PATH} \
    paths.sam3d_results_path=${SAM3D_RESULTS_PATH} \
    paths.result_output_path=${HEAD3D_OUT_PATH} \
    paths.start_mid_end_path=${START_MID_END_PATH} \
    infer.person_list="${PERSON_LIST}" \
    infer.env_list="[all]" \
    # infer.workers=4

echo "🏁 Node ${PBS_SUBREQNO} finished at: $(date)"
# 一个 node 跑一个 person，该 person 下多个 env 并行
#!/usr/bin/env bash
set -u

# Run all compare_with_human_annotation combinations:
#   - mode: fused / single_view
#   - annotation_mode: majority / by_annotator
#
# Usage:
#   bash bash/run_all_compare_with_human_annotation.sh
#   bash bash/run_all_compare_with_human_annotation.sh run.threshold=7.5 run.view=all
#   MAX_JOBS=2 bash bash/run_all_compare_with_human_annotation.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/logs/compare_with_human_annotation"
mkdir -p "$LOG_DIR"
STAMP="$(date +"%Y%m%d_%H%M%S")"
STATUS_DIR="${LOG_DIR}/.${STAMP}_status"
mkdir -p "$STATUS_DIR"

MAX_JOBS="${MAX_JOBS:-4}"
if ! [[ "$MAX_JOBS" =~ ^[1-9][0-9]*$ ]]; then
	echo "Invalid MAX_JOBS=${MAX_JOBS}, fallback to 4"
	MAX_JOBS=4
fi

declare -a CASES=(
	"fused majority"
	"fused by_annotator"
	"single_view majority"
	"single_view by_annotator"
)

EXTRA_OVERRIDES=("$@")

echo "============================================================"
echo "Compare With Human Annotation - All Combinations"
echo "Root: ${ROOT_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Parallel workers: ${MAX_JOBS}"
echo "Start: $(date '+%F %T')"
echo "============================================================"

FAILED=0
declare -a CASE_NAMES=()
declare -A PID_TO_CASE=()

run_case() {
	local mode="$1"
	local annotation_mode="$2"
	local case_name="${mode}__${annotation_mode}"
	local log_file="${LOG_DIR}/${STAMP}_${case_name}.log"
	local status_file="${STATUS_DIR}/${case_name}.exit"

	local -a cmd=(python -m compare_with_human_annotation.main "run.mode=${mode}" "run.annotation_mode=${annotation_mode}")

	if [[ "$mode" == "single_view" ]]; then
		# Keep single_view default explicit for readability; can be overridden by CLI args.
		cmd+=("run.view=all")
	fi

	if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
		cmd+=("${EXTRA_OVERRIDES[@]}")
	fi

	{
		echo "[CASE] ${case_name}"
		echo "[START] $(date '+%F %T')"
		echo "[CMD] ${cmd[*]}"
		echo ""
		"${cmd[@]}"
		exit_code=$?
		echo ""
		echo "[END] $(date '+%F %T')"
		echo "[EXIT] ${exit_code}"
		echo "${exit_code}" > "$status_file"
		exit "$exit_code"
	} > "$log_file" 2>&1 &

	local pid=$!
	PID_TO_CASE["$pid"]="$case_name"
	CASE_NAMES+=("$case_name")

	echo "[LAUNCH] ${case_name} (pid=${pid})"
	echo "[LOG] ${log_file}"
}

current_jobs() {
	jobs -rp | wc -l
}

wait_for_slot() {
	while [[ "$(current_jobs)" -ge "$MAX_JOBS" ]]; do
		wait -n
		sleep 0.1
	done
}

for item in "${CASES[@]}"; do
	mode="${item%% *}"
	annotation_mode="${item##* }"
	wait_for_slot
	echo
	run_case "$mode" "$annotation_mode"
done

echo
echo "Waiting for all background jobs..."
wait

for case_name in "${CASE_NAMES[@]}"; do
	status_file="${STATUS_DIR}/${case_name}.exit"
	if [[ ! -f "$status_file" ]]; then
		echo "[FAIL] ${case_name} (missing status file)"
		FAILED=1
		continue
	fi
	exit_code="$(cat "$status_file")"
	if [[ "$exit_code" != "0" ]]; then
		echo "[FAIL] ${case_name} (exit=${exit_code})"
		FAILED=1
	else
		echo "[OK] ${case_name}"
	fi
done

rm -rf "$STATUS_DIR"

echo
echo "============================================================"
echo "Finished: $(date '+%F %T')"
if [[ $FAILED -ne 0 ]]; then
	echo "Some runs failed. Check logs under: ${LOG_DIR}"
	exit 1
fi
echo "All runs completed successfully."
echo "============================================================"

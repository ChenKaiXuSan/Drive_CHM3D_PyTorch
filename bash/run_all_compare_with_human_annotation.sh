#!/usr/bin/env bash
set -u

# Run all compare_with_human_annotation combinations:
#   - mode: fused / single_view
#   - annotation_mode: majority / by_annotator
#   - threshold: one or more values supplied through THRESHOLDS
#
# Usage:
#   bash bash/run_all_compare_with_human_annotation.sh
#   THRESHOLDS="5.0 7.5 10.0" bash bash/run_all_compare_with_human_annotation.sh run.view=all
#   MAX_JOBS=2 bash bash/run_all_compare_with_human_annotation.sh

# Select what to run here or via environment variables.
# Supported separators: spaces or commas.
# Example:
#   THRESHOLDS="5.0,7.5"
#   RUN_MODES="fused single_view"
#   RUN_ANNOTATION_MODES="majority by_annotator"

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

normalize_list() {
	local value="$1"
	echo "${value//,/ }"
}

RUN_MODES_RAW="$(normalize_list "${RUN_MODES:-fused single_view}")"
RUN_ANNOTATION_MODES_RAW="$(normalize_list "${RUN_ANNOTATION_MODES:-majority by_annotator}")"
THRESHOLDS_RAW="$(normalize_list "${THRESHOLDS:-5.0}")"

read -r -a RUN_MODES_LIST <<< "$RUN_MODES_RAW"
read -r -a RUN_ANNOTATION_MODES_LIST <<< "$RUN_ANNOTATION_MODES_RAW"
read -r -a THRESHOLDS_LIST <<< "$THRESHOLDS_RAW"

if [[ ${#RUN_MODES_LIST[@]} -eq 0 ]]; then
	echo "No modes configured. Set RUN_MODES, for example: RUN_MODES=\"fused single_view\""
	exit 1
fi

if [[ ${#RUN_ANNOTATION_MODES_LIST[@]} -eq 0 ]]; then
	echo "No annotation modes configured. Set RUN_ANNOTATION_MODES, for example: RUN_ANNOTATION_MODES=\"majority by_annotator\""
	exit 1
fi

if [[ ${#THRESHOLDS_LIST[@]} -eq 0 ]]; then
	echo "No thresholds configured. Set THRESHOLDS, for example: THRESHOLDS=\"5.0 7.5 10.0\""
	exit 1
fi

EXTRA_OVERRIDES=("$@")
declare -a EXTRA_OVERRIDES_FILTERED=()

format_threshold_tag() {
	local threshold="$1"
	echo "${threshold//./p}deg"
}

filter_threshold_overrides() {
	local override
	for override in "${EXTRA_OVERRIDES[@]}"; do
		if [[ "$override" == run.threshold=* ]]; then
			continue
		fi
		EXTRA_OVERRIDES_FILTERED+=("$override")
	done
}

filter_threshold_overrides

echo "============================================================"
echo "Compare With Human Annotation - All Combinations"
echo "Root: ${ROOT_DIR}"
echo "Log dir: ${LOG_DIR}"
echo "Parallel workers: ${MAX_JOBS}"
echo "Modes: ${RUN_MODES_LIST[*]}"
echo "Annotation modes: ${RUN_ANNOTATION_MODES_LIST[*]}"
echo "Thresholds: ${THRESHOLDS_LIST[*]}"
echo "Start: $(date '+%F %T')"
echo "============================================================"

FAILED=0
declare -a CASE_NAMES=()
declare -A PID_TO_CASE=()

run_case() {
	local threshold="$1"
	local mode="$2"
	local annotation_mode="$3"
	local threshold_tag
	threshold_tag="$(format_threshold_tag "$threshold")"
	local case_name="${threshold_tag}__${mode}__${annotation_mode}"
	local log_file="${LOG_DIR}/${STAMP}_${case_name}.log"
	local status_file="${STATUS_DIR}/${case_name}.exit"

	local -a cmd=(python -m compare_with_human_annotation.main "run.mode=${mode}" "run.annotation_mode=${annotation_mode}" "run.threshold=${threshold}")

	if [[ "$mode" == "single_view" ]]; then
		# Keep single_view default explicit for readability; can be overridden by CLI args.
		cmd+=("run.view=all")
	fi

	if [[ ${#EXTRA_OVERRIDES_FILTERED[@]} -gt 0 ]]; then
		cmd+=("${EXTRA_OVERRIDES_FILTERED[@]}")
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

for threshold in "${THRESHOLDS_LIST[@]}"; do
	echo
	echo "[THRESHOLD] ${threshold}"
	for mode in "${RUN_MODES_LIST[@]}"; do
		for annotation_mode in "${RUN_ANNOTATION_MODES_LIST[@]}"; do
			wait_for_slot
			echo
			run_case "$threshold" "$mode" "$annotation_mode"
		done
	done
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

#!/usr/bin/env bash
# Batch wrapper that mirrors MotionDecode directory trees while reusing the
# existing, validated single-file DR02 pipeline.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  run_dr02_batch.sh --input-dir DIR [options]

Recursively converts MotionDecode CSV files beneath DIR. DIR must be inside
--dataset-root. Each result is mirrored below output_data and recorded in a
concurrency-safe JSONL manifest.

Options:
  --input-dir DIR       Directory containing input CSV files (required).
  --dataset-root DIR    Dataset root used for mirrored paths
                        (default: dataset/MotionDecode/samples).
  --fps FPS             Source frame rate (default: 120).
  --jobs N              Concurrent conversions (default: min(4, CPU count)).
  --manifest PATH       JSONL manifest path under output_data (default:
                        timestamped path in dr02_data_converter/manifests).
  --overwrite           Rebuild files whose final outputs already exist.
  --limit N             Process at most N discovered files (0 means no limit).
  --max-files N         Alias for --limit.
  --pattern GLOB        find(1) filename glob (default: *.csv; case-insensitive).
  --fail-fast           Stop scheduling new files after the first failed job.
  -h, --help            Show this help text.

Partial final output sets are never reused: without --overwrite they are
recorded as failed. With --overwrite, the pipeline is rerun and its four
staged artifacts atomically replace the corresponding final files.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PIPELINE="${SCRIPT_DIR}/convert_single_motiondecode_to_dr02.sh"
MANIFEST_WRITER="${SCRIPT_DIR}/dr02_batch_manifest.py"
INPUT_DIR=""
DATASET_ROOT="${PROJECT_ROOT}/dataset/MotionDecode/samples"
FPS="120"
JOBS=""
MANIFEST=""
OVERWRITE=false
LIMIT=0
PATTERN="*.csv"
PATTERN_SET=false
FAIL_FAST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-dir) INPUT_DIR="${2:?--input-dir requires a path}"; shift 2 ;;
        --dataset-root) DATASET_ROOT="${2:?--dataset-root requires a path}"; shift 2 ;;
        --fps) FPS="${2:?--fps requires a value}"; shift 2 ;;
        --jobs) JOBS="${2:?--jobs requires a value}"; shift 2 ;;
        --manifest) MANIFEST="${2:?--manifest requires a path}"; shift 2 ;;
        --overwrite) OVERWRITE=true; shift ;;
        --limit|--max-files) LIMIT="${2:?$1 requires a value}"; shift 2 ;;
        --pattern) PATTERN="${2:?--pattern requires a glob}"; PATTERN_SET=true; shift 2 ;;
        --fail-fast) FAIL_FAST=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

[[ -n "$INPUT_DIR" ]] || { echo "[error] --input-dir is required" >&2; usage >&2; exit 2; }
[[ -d "$INPUT_DIR" ]] || { echo "[error] input directory does not exist: $INPUT_DIR" >&2; exit 2; }
[[ -d "$DATASET_ROOT" ]] || { echo "[error] dataset root does not exist: $DATASET_ROOT" >&2; exit 2; }
[[ -x "$PIPELINE" && -f "$MANIFEST_WRITER" ]] || { echo "[error] batch prerequisites are missing" >&2; exit 2; }
[[ "$LIMIT" =~ ^[0-9]+$ ]] || { echo "[error] --limit must be a non-negative integer" >&2; exit 2; }
[[ "$FPS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] --fps must be a positive integer (required by NPZ export)" >&2; exit 2; }
if [[ -n "$JOBS" ]]; then
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "[error] --jobs must be a positive integer" >&2; exit 2; }
else
    CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"
    [[ "$CPU_COUNT" =~ ^[1-9][0-9]*$ ]] || CPU_COUNT=1
    (( CPU_COUNT > 4 )) && CPU_COUNT=4
    JOBS="$CPU_COUNT"
fi

INPUT_DIR_ABS="$(realpath -e "$INPUT_DIR")"
DATASET_ROOT_ABS="$(realpath -e "$DATASET_ROOT")"
if [[ "$INPUT_DIR_ABS" != "$DATASET_ROOT_ABS" && "$INPUT_DIR_ABS" != "$DATASET_ROOT_ABS/"* ]]; then
    echo "[error] --input-dir must be inside --dataset-root" >&2
    exit 2
fi

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
epoch_seconds() { date +%s.%N; }
if [[ -z "$MANIFEST" ]]; then
    MANIFEST="${PROJECT_ROOT}/output_data/dr02_data_converter/manifests/dr02_batch_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
fi
case "$MANIFEST" in
    /*) ;;
    *) MANIFEST="${PROJECT_ROOT}/${MANIFEST}" ;;
esac
MANIFEST="$(realpath -m "$MANIFEST")"
OUTPUT_ROOT="${PROJECT_ROOT}/output_data"
if [[ "$MANIFEST" != "$OUTPUT_ROOT/"* ]]; then
    echo "[error] --manifest must be inside ${OUTPUT_ROOT}" >&2
    exit 2
fi
mkdir -p "$(dirname "$MANIFEST")"
# A global lock keeps two batch invocations from racing the legacy pipeline's
# shared flat output directories. Jobs within this invocation remain parallel.
LOCK_PATH="${OUTPUT_ROOT}/dr02_data_converter/.run_dr02_batch.lock"
mkdir -p "$(dirname "$LOCK_PATH")"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "[error] another DR02 batch is already running (lock: $LOCK_PATH)" >&2
    exit 1
fi
# Ensure an empty run still has a usable, valid (empty) JSONL manifest without
# truncating a caller-supplied manifest from an earlier run.
touch "$MANIFEST"

record() {
    python3 "$MANIFEST_WRITER" record --manifest "$MANIFEST" --run-id "$RUN_ID" \
        --status "$1" --input "$2" --relative-input "$3" \
        --normalized-output "$4" --keypoints-output "$5" --robot-motion-output "$6" --npz-output "$7" \
        --started-at "$8" --finished-at "$9" --duration "${10}" --exit-code "${11}" \
        --failure-reason "${12}" --log-path "${13}"
}

safe_name() {
    local input_abs="$1" input_id slug short_hash
    input_id="$input_abs"
    if [[ "$input_abs" == *"/MotionDecode/samples/"* ]]; then
        input_id="${input_abs#*/MotionDecode/samples/}"
    fi
    slug="$(printf '%s' "${input_id%.csv}" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
    slug="${slug:0:96}"
    [[ -n "$slug" ]] || slug="motiondecode"
    if command -v sha256sum >/dev/null 2>&1; then
        short_hash="$(printf '%s' "$input_abs" | sha256sum | awk '{print substr($1, 1, 12)}')"
    else
        short_hash="$(printf '%s' "$input_abs" | shasum -a 256 | awk '{print substr($1, 1, 12)}')"
    fi
    printf '%s-%s' "$slug" "$short_hash"
}

# Uses the current run_one task's arrays. Only paths whose old final was
# successfully isolated are removed from the legacy-flat location, so a
# failed isolation never deletes an old file that was not moved.
restore_preexisting() {
    local i restore_errors=""
    for i in "${!finals[@]}"; do
        [[ "${isolated[$i]:-0}" == 1 ]] || continue
        if [[ -e "${stages[$i]}" ]] && ! rm -f -- "${stages[$i]}"; then
            restore_errors+="; could not remove generated flat artifact index $i"
            continue
        fi
        if [[ ! -e "${backup_dir}/preexisting-$i" ]]; then
            restore_errors+="; missing preexisting backup index $i"
            continue
        fi
        if ! mv -- "${backup_dir}/preexisting-$i" "${finals[$i]}"; then
            restore_errors+="; could not restore preexisting output index $i"
        elif [[ ! -e "${finals[$i]}" ]]; then
            restore_errors+="; restored output missing at index $i"
        else
            # Make repeated rollback calls safe: after restoration this final
            # is old user data, never a generated flat artifact to remove.
            isolated[$i]=0
        fi
    done
    if [[ -n "$restore_errors" ]]; then
        RESTORE_ERROR="${restore_errors#; }"
        return 1
    fi
    RESTORE_ERROR=""
    return 0
}

run_one() {
    local input="$1" input_abs rel rel_parent unique keyname
    local normalized keypoints motion npz log_path log_dir
    local stage_normalized stage_keypoints stage_motion stage_npz
    local task_stage new_dir backup_dir source_normalized source_keypoints source_motion source_npz
    local started finished started_epoch duration exit_code reason existing_count publish_failed=0 RESTORE_ERROR=""
    local -a finals stages sources backups published isolated
    input_abs="$(realpath -e "$input")"
    rel="${input_abs#"$DATASET_ROOT_ABS"/}"
    [[ "$input_abs" == "$DATASET_ROOT_ABS" ]] && rel="$(basename "$input_abs")"
    rel_parent="$(dirname "$rel")"
    [[ "$rel_parent" == "." ]] && rel_parent=""
    unique="$(safe_name "$input_abs")"
    keyname="${unique}_from_g1"
    normalized="${PROJECT_ROOT}/output_data/dr02_data_converter/normalized/${rel_parent:+$rel_parent/}${unique}_normalized.csv"
    keypoints="${PROJECT_ROOT}/output_data/keypoints/DR02/${rel_parent:+$rel_parent/}${keyname}_keypoints.pkl"
    motion="${PROJECT_ROOT}/output_data/robot_motion/${rel_parent:+$rel_parent/}${keyname}_DR02.csv"
    npz="${PROJECT_ROOT}/output_data/dr02_data_converter/npz/${rel_parent:+$rel_parent/}${keyname}_DR02.npz"
    # Keep one immutable log per batch run.  A stable per-motion directory
    # groups all attempts for the same source CSV, while RUN_ID prevents a
    # later success/failed/skipped attempt from overwriting earlier details.
    log_path="${PROJECT_ROOT}/output_data/dr02_data_converter/logs/${rel_parent:+$rel_parent/}${unique}/${RUN_ID}.log"
    log_dir="$(dirname "$log_path")"
    mkdir -p "$log_dir"
    started="$(timestamp)"; started_epoch="$(epoch_seconds)"
    existing_count=0
    for output in "$normalized" "$keypoints" "$motion" "$npz"; do [[ -e "$output" ]] && ((existing_count+=1)); done
    if (( existing_count == 4 )) && [[ "$OVERWRITE" != true ]]; then
        finished="$(timestamp)"; duration="$(awk "BEGIN {print $(epoch_seconds) - $started_epoch}")"
        printf '[skip] %s (all outputs exist)\n' "$input_abs" >"$log_path"
        record skipped "$input_abs" "$rel" "$normalized" "$keypoints" "$motion" "$npz" "$started" "$finished" "$duration" 0 "" "$log_path"
        return 0
    fi
    if (( existing_count > 0 )) && [[ "$OVERWRITE" != true ]]; then
        finished="$(timestamp)"; duration="$(awk "BEGIN {print $(epoch_seconds) - $started_epoch}")"
        reason="partial final output set exists; rerun with --overwrite to rebuild safely"
        printf '[failed] %s: %s\n' "$input_abs" "$reason" >"$log_path"
        record failed "$input_abs" "$rel" "$normalized" "$keypoints" "$motion" "$npz" "$started" "$finished" "$duration" 1 "$reason" "$log_path"
        return 1
    fi
    mkdir -p "$(dirname "$normalized")" "$(dirname "$keypoints")" "$(dirname "$motion")" "$(dirname "$npz")"
    stage_normalized="${PROJECT_ROOT}/output_data/dr02_data_converter/normalized/${unique}_normalized.csv"
    stage_keypoints="${PROJECT_ROOT}/output_data/keypoints/DR02/${keyname}_keypoints.pkl"
    stage_motion="${PROJECT_ROOT}/output_data/robot_motion/${keyname}_DR02.csv"
    stage_npz="${PROJECT_ROOT}/output_data/dr02_data_converter/npz/${keyname}_DR02.npz"
    task_stage="${PROJECT_ROOT}/output_data/dr02_data_converter/staging/${RUN_ID}/${unique}"
    new_dir="${task_stage}/new"
    backup_dir="${task_stage}/backup"
    # Root-level files have no mirrored parent, so their legacy flat paths are
    # also final paths. Move any prior final aside before invoking the legacy
    # script, then move all newly-created flat artifacts into private staging.
    mkdir -p "$new_dir" "$backup_dir"
    finals=("$normalized" "$keypoints" "$motion" "$npz")
    stages=("$stage_normalized" "$stage_keypoints" "$stage_motion" "$stage_npz")
    isolated=(0 0 0 0)
    for i in "${!finals[@]}"; do
        if [[ "${finals[$i]}" == "${stages[$i]}" && -e "${finals[$i]}" ]]; then
            if ! mv -- "${finals[$i]}" "${backup_dir}/preexisting-$i"; then
                if ! restore_preexisting; then
                    reason="could not isolate root-level final output index $i; rollback incomplete: $RESTORE_ERROR"
                else
                    reason="could not isolate root-level final output index $i; prior outputs restored"
                fi
                finished="$(timestamp)"; duration="$(awk "BEGIN {print $(epoch_seconds) - $started_epoch}")"
                record failed "$input_abs" "$rel" "$normalized" "$keypoints" "$motion" "$npz" "$started" "$finished" "$duration" 1 "$reason" "$log_path"
                return 1
            fi
            isolated[$i]=1
        fi
    done
    set +e
    "$PIPELINE" --input "$input_abs" --fps "$FPS" --no-viewer --overwrite >"$log_path" 2>&1
    exit_code=$?
    set -e
    if (( exit_code == 0 )) && [[ -f "$stage_normalized" && -f "$stage_keypoints" && -f "$stage_motion" && -f "$stage_npz" ]]; then
        # Isolate every generated artifact first; from here publication never
        # depends on the legacy script's shared flat output paths.
        sources=("${new_dir}/0" "${new_dir}/1" "${new_dir}/2" "${new_dir}/3")
        for i in "${!stages[@]}"; do
            if ! mv -- "${stages[$i]}" "${sources[$i]}"; then
                publish_failed=1; reason="could not move generated artifact into private staging"; break
            fi
        done
        backups=("${backup_dir}/0" "${backup_dir}/1" "${backup_dir}/2" "${backup_dir}/3")
        published=()
        if (( publish_failed == 0 )); then
            for i in "${!finals[@]}"; do
                [[ -e "${finals[$i]}" ]] || continue
                if ! mv -- "${finals[$i]}" "${backups[$i]}"; then
                    publish_failed=1; reason="could not back up existing final output"; break
                fi
            done
        fi
        if (( publish_failed == 0 )); then
            for i in "${!finals[@]}"; do
                if mv -- "${sources[$i]}" "${finals[$i]}"; then
                    published+=("${finals[$i]}")
                else
                    publish_failed=1; reason="could not publish staged output"; break
                fi
            done
        fi
        if (( publish_failed == 0 )); then
            rm -f -- "${backups[@]}"
            for i in "${!finals[@]}"; do rm -f -- "${backup_dir}/preexisting-$i"; done
            rmdir "$backup_dir" "$new_dir" "$task_stage" 2>/dev/null || true
            finished="$(timestamp)"; duration="$(awk "BEGIN {print $(epoch_seconds) - $started_epoch}")"
            record success "$input_abs" "$rel" "$normalized" "$keypoints" "$motion" "$npz" "$started" "$finished" "$duration" 0 "" "$log_path"
            return 0
        fi
        # Transaction rollback: remove only files this task published, then
        # restore each same-directory backup. Keep staging for diagnosis.
        for output in "${published[@]}"; do rm -f -- "$output"; done
        for i in "${!finals[@]}"; do
            if [[ -e "${backups[$i]}" ]] && ! mv -- "${backups[$i]}" "${finals[$i]}"; then
                reason+="; could not restore final backup index $i"
            fi
        done
        if ! restore_preexisting; then reason+="; root rollback incomplete: $RESTORE_ERROR"; fi
        exit_code=1
    fi
    if (( exit_code == 0 )); then
        reason="pipeline exited successfully but did not produce all staged artifacts"
    elif [[ -z "${reason:-}" ]]; then
        reason="pipeline failed; see task log"
    fi
    # Restore root-level outputs isolated before a pipeline or collect failure.
    if ! restore_preexisting; then reason+="; root rollback incomplete: $RESTORE_ERROR"; fi
    (( exit_code == 0 )) && exit_code=1
    finished="$(timestamp)"; duration="$(awk "BEGIN {print $(epoch_seconds) - $started_epoch}")"
    record failed "$input_abs" "$rel" "$normalized" "$keypoints" "$motion" "$npz" "$started" "$finished" "$duration" "$exit_code" "$reason" "$log_path"
    return 1
}

RUN_ID="dr02-batch-$(date -u +%Y%m%dT%H%M%SZ)-$$"
if [[ "$PATTERN_SET" == true ]]; then
    mapfile -d '' FILES < <(find "$INPUT_DIR_ABS" -type f -name "$PATTERN" -print0 | sort -z)
else
    mapfile -d '' FILES < <(find "$INPUT_DIR_ABS" -type f -iname '*.csv' -print0 | sort -z)
fi
if (( LIMIT > 0 && ${#FILES[@]} > LIMIT )); then FILES=("${FILES[@]:0:LIMIT}"); fi
echo "[info] discovered ${#FILES[@]} CSV file(s); jobs=$JOBS; manifest=$MANIFEST"

failed=0; scheduled=0; active=0
declare -a PIDS=()
wait_for_one() {
    local completed_pid="" wait_status=0 index
    # Bash 5.1+ reports whichever child finishes first. This makes fail-fast
    # react to a quick failure even while an earlier PID is still running.
    set +e
    wait -n -p completed_pid "${PIDS[@]}"
    wait_status=$?
    set -e
    for index in "${!PIDS[@]}"; do
        if [[ "${PIDS[$index]}" == "$completed_pid" ]]; then
            unset 'PIDS[index]'
            PIDS=("${PIDS[@]}")
            break
        fi
    done
    (( wait_status == 0 )) || failed=1
    ((active-=1)) || true
}
for input in "${FILES[@]}"; do
    if [[ "$FAIL_FAST" == true && "$failed" == 1 ]]; then break; fi
    run_one "$input" & PIDS+=("$!"); ((active+=1)); ((scheduled+=1))
    if (( active >= JOBS )); then wait_for_one; fi
done
while (( active > 0 )); do wait_for_one; done

success_count="$(python3 "$MANIFEST_WRITER" summary --manifest "$MANIFEST" --run-id "$RUN_ID")"
echo "[summary] $success_count; manifest=$MANIFEST"
[[ "$failed" == 0 ]]

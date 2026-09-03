#!/usr/bin/env bash
# Concurrent, process-isolated direct MotionDecode -> DR02 NPZ batch runner.
set -euo pipefail

usage() { cat <<'EOF'
Usage: run_dr02_batch.sh --input-dir DIR [options]
  --dataset-root DIR  Root used to mirror source paths (default: dataset/MotionDecode/samples)
  --fps FPS           Source frame rate (default: 120)
  --jobs N            Independent converter processes (default: min(4, CPUs))
  --manifest PATH     JSONL result manifest under output_data
  --overwrite         Replace final NPZ files
  --limit N           Convert at most N files (0: all)
  --max-files N       Alias for --limit
  --pattern GLOB      Input filename glob (default: *.csv)
  --fail-fast         Stop scheduling after a failed process
EOF
}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DIRECT="${DIRECT_CONVERTER:-$SCRIPT_DIR/direct_motiondecode_to_dr02.py}"; WRITER="$SCRIPT_DIR/dr02_batch_manifest.py"
INPUT_DIR=""; DATASET_ROOT="$ROOT/dataset/MotionDecode/samples"; FPS=120; JOBS=""; MANIFEST=""; OVERWRITE=false; LIMIT=0; PATTERN='*.csv'; FAIL_FAST=false
while [[ $# -gt 0 ]]; do case "$1" in
 --input-dir) INPUT_DIR="${2:?}"; shift 2;; --dataset-root) DATASET_ROOT="${2:?}"; shift 2;; --fps) FPS="${2:?}"; shift 2;; --jobs) JOBS="${2:?}"; shift 2;; --manifest) MANIFEST="${2:?}"; shift 2;; --overwrite) OVERWRITE=true; shift;; --limit|--max-files) LIMIT="${2:?}"; shift 2;; --pattern) PATTERN="${2:?}"; shift 2;; --fail-fast) FAIL_FAST=true; shift;; -h|--help) usage; exit 0;; *) echo "[error] unknown argument: $1" >&2; exit 2;; esac; done
[[ -n "$INPUT_DIR" && -d "$INPUT_DIR" && -d "$DATASET_ROOT" ]] || { echo '[error] valid --input-dir and --dataset-root are required' >&2; exit 2; }
[[ "$FPS" =~ ^[1-9][0-9]*$ && "$LIMIT" =~ ^[0-9]+$ ]] || { echo '[error] fps must be positive and limit non-negative integers' >&2; exit 2; }
[[ -f "$DIRECT" && -f "$WRITER" ]] || { echo '[error] direct converter prerequisites missing' >&2; exit 2; }
if [[ -n "${PYTHON_BIN:-}" ]]; then :
elif [[ -x "$ROOT/.conda/bin/python" ]]; then PYTHON_BIN="$ROOT/.conda/bin/python"
elif [[ -x "$ROOT/conda/bin/python" ]]; then PYTHON_BIN="$ROOT/conda/bin/python"
elif command -v python >/dev/null 2>&1; then PYTHON_BIN="$(command -v python)"
else PYTHON_BIN="$(command -v python3)"; fi
[[ -n "$JOBS" ]] || JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"; (( JOBS > 4 )) && JOBS=4
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo '[error] jobs must be positive' >&2; exit 2; }
INPUT_DIR="$(realpath -e "$INPUT_DIR")"; DATASET_ROOT="$(realpath -e "$DATASET_ROOT")"
[[ "$INPUT_DIR" == "$DATASET_ROOT" || "$INPUT_DIR" == "$DATASET_ROOT/"* ]] || { echo '[error] input must be within dataset root' >&2; exit 2; }
OUTPUT_ROOT="$ROOT/output_data/dr02_data_converter"
NPZ_ROOT="$OUTPUT_ROOT/npz"
MANIFEST="${MANIFEST:-$OUTPUT_ROOT/manifests/dr02_batch_$(date -u +%Y%m%dT%H%M%SZ).jsonl}"; [[ "$MANIFEST" = /* ]] || MANIFEST="$ROOT/$MANIFEST"; MANIFEST="$(realpath -m "$MANIFEST")"
[[ "$MANIFEST" == "$ROOT/output_data/"* ]] || { echo "[error] --manifest must be under $ROOT/output_data" >&2; exit 2; }
mkdir -p "$(dirname "$MANIFEST")" "$NPZ_ROOT"; touch "$MANIFEST"

# The direct converter has no shared intermediate artifacts, but a batch-level
# lock still prevents two invocations from racing to publish the same final NPZ.
LOCK_PATH="$OUTPUT_ROOT/.run_dr02_batch.lock"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "[error] another DR02 batch is already running (lock: $LOCK_PATH)" >&2
    exit 1
fi
RUN_ID="dr02-batch-$(date -u +%Y%m%dT%H%M%SZ)-$$"; export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
record() { "$PYTHON_BIN" "$WRITER" record --manifest "$MANIFEST" --run-id "$RUN_ID" --status "$1" --input "$2" --relative-input "$3" --npz-output "$4" --started-at "$5" --finished-at "$6" --duration "$7" --exit-code "$8" --failure-reason "$9" --log-path "${10}"; }
run_one() {
    local input="$1" rel out log start end start_epoch end_epoch elapsed rc
    local -a converter_args
    rel="${input#$DATASET_ROOT/}"
    out="$NPZ_ROOT/${rel%.[cC][sS][vV]}_from_g1_DR02.npz"
    log="$OUTPUT_ROOT/logs/${rel}.log"
    mkdir -p "$(dirname "$out")" "$(dirname "$log")"
    start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"; start_epoch="$(date +%s.%N)"
    if [[ -e "$out" && "$OVERWRITE" != true ]]; then
        record skipped "$input" "$rel" "$out" "$start" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" 0 0 '' "$log"
        return 0
    fi
    converter_args=("$PYTHON_BIN" "$DIRECT" --input "$input" --output "$out" --fps "$FPS")
    [[ "$OVERWRITE" == true ]] && converter_args+=(--overwrite)
    set +e; "${converter_args[@]}" >"$log" 2>&1; rc=$?; set -e
    end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    end_epoch="$(date +%s.%N)"
    elapsed=$(awk -v start="$start_epoch" -v end="$end_epoch" 'BEGIN {print end - start}')
    if (( rc == 0 )) && [[ -s "$out" ]]; then
        record success "$input" "$rel" "$out" "$start" "$end" "$elapsed" 0 '' "$log"
    else
        record failed "$input" "$rel" "$out" "$start" "$end" "$elapsed" "${rc:-1}" 'converter failed; see log' "$log"
        return 1
    fi
}
mapfile -d '' FILES < <(find "$INPUT_DIR" -type f -iname "$PATTERN" -print0 | sort -z)
if (( LIMIT > 0 && ${#FILES[@]} > LIMIT )); then FILES=("${FILES[@]:0:LIMIT}"); fi
echo "[info] discovered ${#FILES[@]} file(s); jobs=$JOBS; manifest=$MANIFEST"
active=0; failed=0; PIDS=()
cleanup_interrupted_batch() {
    local pid
    trap - INT TERM HUP
    for pid in "${PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
    for pid in "${PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
    # A killed Python process cannot run its finally block.  With the batch
    # lock held, these files can only belong to this interrupted invocation.
    find "$NPZ_ROOT" -type f -name '.*.tmp.npz' -delete 2>/dev/null || true
    exit 130
}
trap cleanup_interrupted_batch INT TERM HUP
wait_one() {
    local completed_pid='' rc=0 index
    set +e
    wait -n -p completed_pid "${PIDS[@]}"
    rc=$?
    set -e
    for index in "${!PIDS[@]}"; do
        if [[ "${PIDS[$index]}" == "$completed_pid" ]]; then
            unset 'PIDS[index]'
            PIDS=("${PIDS[@]}")
            break
        fi
    done
    ((active-=1)) || true
    ((rc==0)) || failed=1
}
for f in "${FILES[@]}"; do [[ "$FAIL_FAST" == true && $failed == 1 ]] && break; run_one "$f" & PIDS+=("$!"); ((active+=1)); ((active>=JOBS)) && wait_one; done
while ((active)); do wait_one; done
"$PYTHON_BIN" "$WRITER" summary --manifest "$MANIFEST" --run-id "$RUN_ID"; ((failed==0))

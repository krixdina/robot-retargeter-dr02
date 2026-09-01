#!/usr/bin/env bash
# Convert one MotionDecode G1 CSV through the isolated DR02 two-stage IK test path.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  convert_single_motiondecode_to_dr02_two_stage_optimization.sh --input <MotionDecode.csv> [--fps 120] [--output-name NAME] [--no-compare|--no-viewer] [--overwrite]

Options:
  --input PATH       MotionDecode G1 CSV to convert (required).
  --fps FPS          Source/capture frame rate for all stages (default: 120).
  --output-name NAME Output stem; defaults to the input CSV stem.
  --no-compare       Skip the final side-by-side MuJoCo G1/DR02 viewer.
  --no-viewer        Deprecated alias for --no-compare.
  --overwrite        Replace outputs in this isolated two-stage test directory.
  -h, --help         Show this help text.

The final comparison viewer is enabled by default.  This script uses the
separate DR02_two_stage_optimization_test.yaml, so normal DR02.yaml behavior
is never changed.  Its two-stage mode is a first-frame-only initialization:
coarse IK and a fine warm-start run only at frame 0; frame 1+ use the normal
temporal and full-frame velocity-constrained IK path.  Set PYTHON_BIN to
select a Python interpreter explicitly.
EOF
}

INPUT=""; FPS="120"; OUTPUT_NAME=""; NO_COMPARE=false; OVERWRITE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input) [[ $# -ge 2 ]] || { echo "[error] --input requires a path" >&2; exit 2; }; INPUT="$2"; shift 2 ;;
        --fps) [[ $# -ge 2 ]] || { echo "[error] --fps requires a value" >&2; exit 2; }; FPS="$2"; shift 2 ;;
        --output-name) [[ $# -ge 2 ]] || { echo "[error] --output-name requires a value" >&2; exit 2; }; OUTPUT_NAME="$2"; shift 2 ;;
        --no-compare) NO_COMPARE=true; shift ;;
        --no-viewer) echo "[warn] --no-viewer is deprecated; use --no-compare" >&2; NO_COMPARE=true; shift ;;
        --overwrite) OVERWRITE=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$INPUT" ]] || { echo "[error] --input is required" >&2; usage >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"
[[ -f "$INPUT" ]] || { echo "[error] input CSV not found: $INPUT" >&2; exit 1; }
[[ "${INPUT##*.}" == "csv" || "${INPUT##*.}" == "CSV" ]] || { echo "[error] input must be a CSV" >&2; exit 2; }
[[ "$FPS" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ && "$FPS" != "0" && "$FPS" != "0.0" ]] || { echo "[error] --fps must be positive" >&2; exit 2; }

INPUT_ABS="$(realpath -e "$INPUT")"
if [[ -z "$OUTPUT_NAME" ]]; then OUTPUT_NAME="$(basename "${INPUT_ABS%.*}")"; fi
[[ -n "$OUTPUT_NAME" && "$OUTPUT_NAME" != */* && "$OUTPUT_NAME" != . && "$OUTPUT_NAME" != .. ]] || { echo "[error] --output-name must be one filename component" >&2; exit 2; }

if [[ -n "${PYTHON_BIN:-}" ]]; then :
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -x "/home/krixdina/miniconda3/envs/robot_retargeter/bin/python" ]]; then PYTHON_BIN="/home/krixdina/miniconda3/envs/robot_retargeter/bin/python"
elif command -v python3 >/dev/null 2>&1; then PYTHON_BIN="$(command -v python3)"
else echo "[error] no Python interpreter found; set PYTHON_BIN" >&2; exit 127; fi

SOURCE_CONFIG="config/robot/g1.yaml"
KEYPOINT_CONFIG="config/robot/DR02.yaml"
TWO_STAGE_CONFIG="config/robot/DR02_two_stage_optimization_test.yaml"
CONFIG_STEM="DR02_two_stage_optimization_test"
ADAPTER="${SCRIPT_DIR}/motiondecode_adapter.py"
NPZ_EXPORTER="${SCRIPT_DIR}/dr02_npz_exporter.py"
COMPARER="${SCRIPT_DIR}/compare_g1_dr02_mujoco.py"
for file in "$ADAPTER" "$NPZ_EXPORTER" "$COMPARER" scripts/robot_replay.py scripts/robot_retarget.py "$SOURCE_CONFIG" "$KEYPOINT_CONFIG" "$TWO_STAGE_CONFIG"; do
    [[ -f "$file" ]] || { echo "[error] required file not found: $file" >&2; exit 1; }
done

KEYPOINTS_NAME="${OUTPUT_NAME}_from_g1"
NORMALIZED_DIR="output_data/dr02_data_converter/two_stage_optimization_test/normalized"
NPZ_DIR="output_data/dr02_data_converter/two_stage_optimization_test/npz"
NORMALIZED_PATH="${NORMALIZED_DIR}/${OUTPUT_NAME}_normalized.csv"
KEYPOINTS_PATH="output_data/keypoints/${CONFIG_STEM}/${KEYPOINTS_NAME}_keypoints.pkl"
ROBOT_MOTION_PATH="output_data/robot_motion/${KEYPOINTS_NAME}_${CONFIG_STEM}.csv"
NPZ_PATH="${NPZ_DIR}/${KEYPOINTS_NAME}_${CONFIG_STEM}.npz"
OUTPUTS=("$NORMALIZED_PATH" "$KEYPOINTS_PATH" "$ROBOT_MOTION_PATH" "$NPZ_PATH")
EXISTING=(); for file in "${OUTPUTS[@]}"; do [[ -e "$file" ]] && EXISTING+=("$file"); done
if [[ ${#EXISTING[@]} -gt 0 && "$OVERWRITE" != true ]]; then
    echo "[error] outputs already exist; pass --overwrite:" >&2; printf '  %s\n' "${EXISTING[@]}" >&2; exit 1
fi
mkdir -p "$NORMALIZED_DIR" "$NPZ_DIR" "$(dirname "$KEYPOINTS_PATH")" "$(dirname "$ROBOT_MOTION_PATH")"
if [[ ${#EXISTING[@]} -gt 0 ]]; then rm -f -- "${EXISTING[@]}"; fi

echo "=== DR02 two-stage optimization test pipeline ==="
echo "[1/5] Normalize MotionDecode CSV"
"$PYTHON_BIN" "$ADAPTER" --input "$INPUT_ABS" --output "$NORMALIZED_PATH"
echo "[2/5] Replay G1 motion to DR02 target keypoints"
"$PYTHON_BIN" scripts/robot_replay.py --no-viewer --motion-file "$NORMALIZED_PATH" --source-robot-config "$SOURCE_CONFIG" --target-robot-config "$KEYPOINT_CONFIG" --fps "$FPS" --output-path "$KEYPOINTS_PATH"
echo "[3/5] Retarget with isolated coarse-to-fine IK"
"$PYTHON_BIN" scripts/robot_retarget.py --config "$TWO_STAGE_CONFIG" --keypoints-name "$KEYPOINTS_NAME" --no-render-debug
[[ -f "$ROBOT_MOTION_PATH" ]] || { echo "[error] expected DR02 motion was not produced" >&2; exit 1; }
echo "[4/5] Export DR02 NPZ"
NPZ_ARGS=(--input "$ROBOT_MOTION_PATH" --output "$NPZ_PATH" --fps "$FPS")
[[ "$OVERWRITE" == true ]] && NPZ_ARGS+=(--overwrite)
"$PYTHON_BIN" "$NPZ_EXPORTER" "${NPZ_ARGS[@]}"
if [[ "$NO_COMPARE" == true ]]; then
    echo "[5/5] Skip MuJoCo comparison (--no-compare)"
else
    echo "[5/5] Compare source G1 and retargeted DR02 in MuJoCo"
    "$PYTHON_BIN" "$COMPARER" --source-csv "$INPUT_ABS" --target-csv "$ROBOT_MOTION_PATH" --source-fps "$FPS" --target-fps "$FPS" --render-fps "$FPS"
fi
printf '[done] %s\n' "$NORMALIZED_PATH" "$KEYPOINTS_PATH" "$ROBOT_MOTION_PATH" "$NPZ_PATH"

#!/usr/bin/env bash
# Convert a MotionDecode CSV into a DR02 motion CSV/NPZ in one reproducible run.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  convert_single_motiondecode_to_dr02.sh --input <MotionDecode.csv> [--fps 120] [--output-name NAME] [--no-compare|--no-viewer] [--overwrite]

Options:
  --input PATH       MotionDecode CSV to convert (required).
  --fps FPS          Source/capture frame rate passed through every stage (default: 120).
  --output-name NAME Override the output stem (default: source CSV stem).
                     Batch conversion uses this only for unique internal staging.
  --no-compare       Do not open the final MuJoCo G1/DR02 comparison viewer.
  --no-viewer        Deprecated alias for --no-compare.
  --overwrite        Replace this pipeline's pre-existing outputs for the same input.
  -h, --help         Show this help text.

Set PYTHON_BIN to explicitly select Python. Otherwise the script prefers a
project-local conda interpreter, then python, then python3.
EOF
}

INPUT=""
FPS="120"
OUTPUT_NAME=""
NO_COMPARE=false
OVERWRITE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            [[ $# -ge 2 ]] || { echo "[error] --input requires a path" >&2; exit 2; }
            INPUT="$2"
            shift 2
            ;;
        --fps)
            [[ $# -ge 2 ]] || { echo "[error] --fps requires a value" >&2; exit 2; }
            FPS="$2"
            shift 2
            ;;
        --output-name)
            [[ $# -ge 2 ]] || { echo "[error] --output-name requires a value" >&2; exit 2; }
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --no-compare)
            NO_COMPARE=true
            shift
            ;;
        --no-viewer)
            echo "[warn] --no-viewer is deprecated; use --no-compare" >&2
            NO_COMPARE=true
            shift
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[error] unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "${INPUT}" ]] || { echo "[error] --input is required" >&2; usage >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ ! -f "${INPUT}" ]]; then
    echo "[error] input CSV not found: ${INPUT}" >&2
    exit 1
fi
if [[ "${INPUT##*.}" != "csv" && "${INPUT##*.}" != "CSV" ]]; then
    echo "[error] input must be a .csv file: ${INPUT}" >&2
    exit 1
fi
if ! [[ "${FPS}" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || [[ "${FPS}" == "0" || "${FPS}" == "0.0" ]]; then
    echo "[error] --fps must be a positive numeric value: ${FPS}" >&2
    exit 2
fi

# Resolve relative inputs before changing their name. realpath -e also prevents
# a symlinked/non-existent input from accidentally being treated as an output.
INPUT_ABS="$(realpath -e "${INPUT}")"

if [[ -z "${OUTPUT_NAME}" ]]; then
    SOURCE_FILENAME="$(basename "${INPUT_ABS}")"
    OUTPUT_NAME="${SOURCE_FILENAME%.*}"
fi
if [[ -z "${OUTPUT_NAME}" || "${OUTPUT_NAME}" == "." || "${OUTPUT_NAME}" == ".." || "${OUTPUT_NAME}" == */* ]]; then
    echo "[error] --output-name must be one non-empty filename component" >&2
    exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    : # User override has the highest priority.
elif [[ -x "${PROJECT_ROOT}/.conda/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.conda/bin/python"
elif [[ -x "${PROJECT_ROOT}/conda/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/conda/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/$(basename "${PROJECT_ROOT}")/bin/python" ]]; then
    # Conventional per-project Conda environment, e.g. envs/robot_retargeter.
    PYTHON_BIN="${HOME}/miniconda3/envs/$(basename "${PROJECT_ROOT}")/bin/python"
elif [[ -x "${HOME}/anaconda3/envs/$(basename "${PROJECT_ROOT}")/bin/python" ]]; then
    PYTHON_BIN="${HOME}/anaconda3/envs/$(basename "${PROJECT_ROOT}")/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "[error] no Python interpreter found; set PYTHON_BIN explicitly" >&2
    exit 127
fi
if ! "${PYTHON_BIN}" --version >/dev/null 2>&1; then
    echo "[error] PYTHON_BIN is not a working Python interpreter: ${PYTHON_BIN}" >&2
    exit 127
fi

ADAPTER="${SCRIPT_DIR}/motiondecode_adapter.py"
NPZ_EXPORTER="${SCRIPT_DIR}/dr02_npz_exporter.py"
COMPARER="${SCRIPT_DIR}/compare_g1_dr02_mujoco.py"
for required_file in "${ADAPTER}" "${NPZ_EXPORTER}" "${COMPARER}" scripts/robot_replay.py scripts/robot_retarget.py config/robot/g1.yaml config/robot/DR02.yaml; do
    [[ -f "${required_file}" ]] || { echo "[error] required file not found: ${required_file}" >&2; exit 1; }
done

SAFE_NAME="${OUTPUT_NAME}"
KEYPOINTS_NAME="${SAFE_NAME}_from_g1"

NORMALIZED_DIR="output_data/dr02_data_converter/normalized"
NPZ_DIR="output_data/dr02_data_converter/npz"
KEYPOINTS_PATH="output_data/keypoints/DR02/${KEYPOINTS_NAME}_keypoints.pkl"
ROBOT_MOTION_PATH="output_data/robot_motion/${KEYPOINTS_NAME}_DR02.csv"
NORMALIZED_PATH="${NORMALIZED_DIR}/${SAFE_NAME}_normalized.csv"
NPZ_PATH="${NPZ_DIR}/${KEYPOINTS_NAME}_DR02.npz"

OUTPUTS=("${NORMALIZED_PATH}" "${KEYPOINTS_PATH}" "${ROBOT_MOTION_PATH}" "${NPZ_PATH}")
EXISTING=()
for output in "${OUTPUTS[@]}"; do
    [[ -e "${output}" ]] && EXISTING+=("${output}")
done
if [[ ${#EXISTING[@]} -gt 0 && "${OVERWRITE}" != true ]]; then
    echo "[error] output already exists (pass --overwrite to replace it):" >&2
    printf '  %s\n' "${EXISTING[@]}" >&2
    exit 1
fi

mkdir -p "${NORMALIZED_DIR}" "${NPZ_DIR}" "$(dirname "${KEYPOINTS_PATH}")" "$(dirname "${ROBOT_MOTION_PATH}")"
if [[ ${#EXISTING[@]} -gt 0 ]]; then
    echo "[info] removing ${#EXISTING[@]} existing pipeline output(s)"
    rm -f -- "${EXISTING[@]}"
fi

echo "=== DR02 MotionDecode pipeline ==="
echo "[info] input: ${INPUT_ABS}"
echo "[info] fps: ${FPS}"
echo "[info] output name: ${SAFE_NAME}"
echo "[info] motion name: ${KEYPOINTS_NAME}"

echo "[1/5] Normalize MotionDecode CSV"
"${PYTHON_BIN}" "${ADAPTER}" --input "${INPUT_ABS}" --output "${NORMALIZED_PATH}"

echo "[2/5] Replay g1 motion to DR02 keypoints"
"${PYTHON_BIN}" scripts/robot_replay.py \
    --no-viewer \
    --motion-file "${NORMALIZED_PATH}" \
    --source-robot-config config/robot/g1.yaml \
    --target-robot-config config/robot/DR02.yaml \
    --fps "${FPS}" \
    --output-path "${KEYPOINTS_PATH}"

echo "[3/5] Retarget DR02 motion"
"${PYTHON_BIN}" scripts/robot_retarget.py \
    --config config/robot/DR02.yaml \
    --keypoints-name "${KEYPOINTS_NAME}" \
    --no-render-debug

[[ -f "${ROBOT_MOTION_PATH}" ]] || { echo "[error] expected DR02 motion was not produced: ${ROBOT_MOTION_PATH}" >&2; exit 1; }

echo "[4/5] Export DR02 NPZ"
"${PYTHON_BIN}" "${NPZ_EXPORTER}" --input "${ROBOT_MOTION_PATH}" --output "${NPZ_PATH}" --fps "${FPS}"

if [[ "${NO_COMPARE}" == true ]]; then
    echo "[5/5] Skip G1/DR02 comparison (--no-compare)"
else
    echo "[5/5] Compare source G1 and retargeted DR02 motion"
    "${PYTHON_BIN}" "${COMPARER}" \
        --source-csv "${INPUT_ABS}" \
        --target-csv "${ROBOT_MOTION_PATH}" \
        --source-fps "${FPS}" \
        --target-fps "${FPS}" \
        --render-fps "${FPS}"
fi

echo "[done] normalized CSV: ${NORMALIZED_PATH}"
echo "[done] DR02 keypoints: ${KEYPOINTS_PATH}"
echo "[done] DR02 motion CSV: ${ROBOT_MOTION_PATH}"
echo "[done] DR02 NPZ: ${NPZ_PATH}"

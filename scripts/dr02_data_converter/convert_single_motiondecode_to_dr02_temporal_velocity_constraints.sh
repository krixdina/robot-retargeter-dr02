#!/usr/bin/env bash
# Convert one MotionDecode G1 CSV with the DR02 temporal/velocity test constraints.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  convert_single_motiondecode_to_dr02_temporal_velocity_constraints.sh --input <MotionDecode.csv> [--fps 120] [--output-name NAME] [--no-viewer] [--overwrite]

Options:
  --input PATH       MotionDecode CSV to convert (required).
  --fps FPS          Source/capture frame rate used by every stage (default: 120).
  --output-name NAME Override the output stem (default: source CSV stem).
  --no-viewer        Do not open the final MuJoCo visualization window.
  --overwrite        Replace this test pipeline's existing outputs.
  -h, --help         Show this help text.

This test pipeline first uses config/robot/DR02.yaml to build target keypoints,
then uses config/robot/DR02_temporal_constraints_test.yaml for IK. The latter
enables the temporal continuity regularizer and the inter-frame joint velocity
constraint without changing the normal DR02.yaml workflow.

Set PYTHON_BIN to explicitly select Python. Otherwise the script prefers the
active Conda environment, a conventional project Conda environment, then
python/python3 from PATH.
EOF
}

INPUT=""
FPS="120"
OUTPUT_NAME=""
NO_VIEWER=false
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
        --no-viewer)
            NO_VIEWER=true
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
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/$(basename "${PROJECT_ROOT}")/bin/python" ]]; then
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

SOURCE_CONFIG="config/robot/g1.yaml"
KEYPOINT_CONFIG="config/robot/DR02.yaml"
CONSTRAINT_CONFIG="config/robot/DR02_temporal_constraints_test.yaml"
CONSTRAINT_ROBOT="$(basename "${CONSTRAINT_CONFIG}" .yaml)"
ADAPTER="${SCRIPT_DIR}/motiondecode_adapter.py"
NPZ_EXPORTER="${SCRIPT_DIR}/dr02_npz_exporter.py"

for required_file in \
    "${ADAPTER}" \
    "${NPZ_EXPORTER}" \
    scripts/robot_replay.py \
    scripts/robot_retarget.py \
    scripts/multi_robot_visualize.py \
    "${SOURCE_CONFIG}" \
    "${KEYPOINT_CONFIG}" \
    "${CONSTRAINT_CONFIG}"; do
    [[ -f "${required_file}" ]] || { echo "[error] required file not found: ${required_file}" >&2; exit 1; }
done

SAFE_NAME="${OUTPUT_NAME}"
KEYPOINTS_NAME="${SAFE_NAME}_from_g1"
NORMALIZED_DIR="output_data/dr02_data_converter/temporal_constraints_test/normalized"
NPZ_DIR="output_data/dr02_data_converter/temporal_constraints_test/npz"
KEYPOINTS_PATH="output_data/keypoints/${CONSTRAINT_ROBOT}/${KEYPOINTS_NAME}_keypoints.pkl"
ROBOT_MOTION_PATH="output_data/robot_motion/${KEYPOINTS_NAME}_${CONSTRAINT_ROBOT}.csv"
NORMALIZED_PATH="${NORMALIZED_DIR}/${SAFE_NAME}_normalized.csv"
NPZ_PATH="${NPZ_DIR}/${KEYPOINTS_NAME}_${CONSTRAINT_ROBOT}.npz"

OUTPUTS=("${NORMALIZED_PATH}" "${KEYPOINTS_PATH}" "${ROBOT_MOTION_PATH}" "${NPZ_PATH}")
EXISTING=()
for output in "${OUTPUTS[@]}"; do
    [[ -e "${output}" ]] && EXISTING+=("${output}")
done
if [[ ${#EXISTING[@]} -gt 0 && "${OVERWRITE}" != true ]]; then
    echo "[error] test output already exists (pass --overwrite to replace it):" >&2
    printf '  %s\n' "${EXISTING[@]}" >&2
    exit 1
fi

mkdir -p "${NORMALIZED_DIR}" "${NPZ_DIR}" "$(dirname "${KEYPOINTS_PATH}")" "$(dirname "${ROBOT_MOTION_PATH}")"
if [[ ${#EXISTING[@]} -gt 0 ]]; then
    echo "[info] removing ${#EXISTING[@]} existing test pipeline output(s)"
    rm -f -- "${EXISTING[@]}"
fi

echo "=== DR02 temporal constraints test pipeline ==="
echo "[info] Python: ${PYTHON_BIN}"
echo "[info] input: ${INPUT_ABS}"
echo "[info] fps: ${FPS}"
echo "[info] output name: ${SAFE_NAME}"
echo "[info] keypoint config: ${KEYPOINT_CONFIG}"
echo "[info] constrained IK config: ${CONSTRAINT_CONFIG}"

echo "[1/5] Normalize MotionDecode CSV"
"${PYTHON_BIN}" "${ADAPTER}" \
    --input "${INPUT_ABS}" \
    --output "${NORMALIZED_PATH}"

echo "[2/5] Replay G1 motion to DR02 target keypoints"
"${PYTHON_BIN}" scripts/robot_replay.py \
    --no-viewer \
    --motion-file "${NORMALIZED_PATH}" \
    --source-robot-config "${SOURCE_CONFIG}" \
    --target-robot-config "${KEYPOINT_CONFIG}" \
    --fps "${FPS}" \
    --output-path "${KEYPOINTS_PATH}"

echo "[3/5] Retarget with temporal and inter-frame velocity constraints"
"${PYTHON_BIN}" scripts/robot_retarget.py \
    --config "${CONSTRAINT_CONFIG}" \
    --keypoints-name "${KEYPOINTS_NAME}" \
    --no-render-debug

[[ -f "${ROBOT_MOTION_PATH}" ]] || {
    echo "[error] expected constrained DR02 motion was not produced: ${ROBOT_MOTION_PATH}" >&2
    exit 1
}

echo "[4/5] Export constrained DR02 NPZ"
NPZ_ARGS=(
    --input "${ROBOT_MOTION_PATH}"
    --output "${NPZ_PATH}"
    --fps "${FPS}"
)
if [[ "${OVERWRITE}" == true ]]; then
    NPZ_ARGS+=(--overwrite)
fi
"${PYTHON_BIN}" "${NPZ_EXPORTER}" "${NPZ_ARGS[@]}"

if [[ "${NO_VIEWER}" == true ]]; then
    echo "[5/5] Skip constrained DR02 visualization (--no-viewer)"
else
    echo "[5/5] Visualize constrained DR02 motion"
    "${PYTHON_BIN}" scripts/multi_robot_visualize.py \
        --motion "${KEYPOINTS_NAME}" \
        --robots "${CONSTRAINT_ROBOT}" \
        --source_fps "${FPS}" \
        --render_fps "${FPS}"
fi

echo "[done] normalized CSV: ${NORMALIZED_PATH}"
echo "[done] constrained keypoints: ${KEYPOINTS_PATH}"
echo "[done] constrained DR02 motion CSV: ${ROBOT_MOTION_PATH}"
echo "[done] constrained DR02 NPZ: ${NPZ_PATH}"

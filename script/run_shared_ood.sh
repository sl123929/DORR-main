#!/usr/bin/env bash
# Reuse the existing frozen representation split; train and calibrate a new shared detector.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
DATA_DIR="${DATA_DIR:?Set DATA_DIR to the directory containing id_train.npz and id_calibration.npz}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR to a new detector output directory}"
TASK_SAMPLING="${TASK_SAMPLING:-random}"
stable_args=()
if [[ "${STABLE_SCORE_NOISE:-0}" == 1 ]]; then stable_args=(--stable-score-noise); fi
for name in id_train.npz id_calibration.npz; do
  [[ -f "$DATA_DIR/$name" ]] || { echo "Missing dataset: $DATA_DIR/$name" >&2; exit 2; }
done
for name in detector_uncalibrated.pt detector.pt; do
  [[ ! -e "$OUTPUT_DIR/$name" ]] || { echo "Already exists: $OUTPUT_DIR/$name; choose a new OUTPUT_DIR." >&2; exit 2; }
done
mkdir -p "$OUTPUT_DIR"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3 BLIS_NUM_THREADS=3 VECLIB_MAXIMUM_THREADS=3
export ROLeR_OOD_CPU_THREADS=3 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
python run_ood_detector.py train \
  --dataset "$DATA_DIR/id_train.npz" \
  --checkpoint "$OUTPUT_DIR/detector_uncalibrated.pt" \
  --architecture shared --task-sampling "$TASK_SAMPLING" \
  --device cuda --seed 0 --epochs 100 --batch-size 256 \
  --learning-rate 3e-4 --hidden-dim 256 --time-embed-dim 16 \
  --num-hidden-layers 4 --score-levels 1
python run_ood_detector.py calibrate \
  --dataset "$DATA_DIR/id_calibration.npz" \
  --reference-dataset "$DATA_DIR/id_train.npz" \
  --checkpoint "$OUTPUT_DIR/detector_uncalibrated.pt" \
  --output-checkpoint "$OUTPUT_DIR/detector.pt" \
  --percentiles 90 95 99 "${stable_args[@]}" --device cuda
echo "Shared detector: $OUTPUT_DIR/detector.pt"
echo "DORR calibration dataset: $DATA_DIR/id_calibration.npz"

#!/usr/bin/env bash
# Usage: GPU_ID=0 bash script/run_dorr_cross_env.sh prepare|train|all ENV INITIAL_CHECKPOINT [SEED]
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_root"
mode="${1:-}"
env_name="${2:-}"
initial="${3:-}"
seed="${4:-0}"
source_root="$(cd -- "${SOURCE_ROOT:-$project_root}" && pwd -P)"
output_root="${OUTPUT_ROOT:-$(dirname "$project_root")/DORR_runs}"
output_root="$(realpath -m "$output_root")"

if [[ "$mode" != prepare && "$mode" != train && "$mode" != all ]]; then
  echo "Usage: GPU_ID=0 bash script/run_dorr_cross_env.sh prepare|train|all KuaiEnv-v0|KuaiRand-v0|CoatEnv-v0|YahooEnv-v0 /path/to/initial.pt [seed]" >&2
  exit 2
fi
if [[ "$output_root" == "$source_root" || "$output_root" == "$source_root/"* ]]; then
  echo "The output root must be outside the source root." >&2
  exit 2
fi
mkdir -p "$output_root"
[[ -f "$initial" ]] || { echo "Missing ROLeR checkpoint: $initial" >&2; exit 2; }
initial="$(realpath "$initial")"
[[ "$initial" == "$source_root/"* ]] || {
  echo "The initial checkpoint must be inside the source root: $source_root" >&2; exit 2;
}
[[ "$seed" =~ ^[0-9]+$ ]] || { echo "Seed must be a nonnegative integer." >&2; exit 2; }

case "$env_name" in
  KuaiEnv-v0)
    tracker=att; window=3; variance=0.05; entropy=5; kr=30; ku=30 ;;
  KuaiRand-v0)
    tracker=avg; window=3; variance=0.01; entropy=0.05; kr=150; ku=150 ;;
  CoatEnv-v0)
    tracker=att; window=3; variance=0.005; entropy=1; kr=15; ku=15 ;;
  YahooEnv-v0)
    tracker=att; window=5; variance=0.005; entropy=1; kr=20; ku=20 ;;
  *) echo "Unsupported environment: $env_name" >&2; exit 2 ;;
esac

environment_root="$source_root/environments"
case "$env_name" in
  KuaiEnv-v0)
    data_files=(KuaiRec/data/big_matrix_processed.csv
      KuaiRec/data/small_matrix_processed.csv
      KuaiRec/data/user_features.csv KuaiRec/data/item_categories.csv) ;;
  KuaiRand-v0)
    data_files=(
      KuaiRand_Pure/MF_results_GT/kuairand_is_click.csv
      KuaiRand_Pure/data/train_processed.csv
      KuaiRand_Pure/data/test_processed.csv
      KuaiRand_Pure/data/log_standard_4_22_to_5_08_pure.csv
      KuaiRand_Pure/data/user_features_pure.csv
      KuaiRand_Pure/data/video_features_basic_pure.csv
    ) ;;
  CoatEnv-v0)
    data_files=(coat/train.ascii coat/test.ascii
      coat/user_item_features/user_features.ascii
      coat/user_item_features/item_features.ascii
      RL4Rec/data/coat_pseudoGT_ratingM.ascii) ;;
  YahooEnv-v0)
    data_files=(YahooR3/ydata-ymusic-rating-study-v1_0-train.txt
      YahooR3/ydata-ymusic-rating-study-v1_0-test.txt
      RL4Rec/data/yahoo_pseudoGT_ratingM.ascii) ;;
esac
for relative_path in "${data_files[@]}"; do
  [[ -f "$environment_root/$relative_path" ]] || {
    echo "Missing source dataset: $environment_root/$relative_path" >&2; exit 2;
  }
done
echo "Dataset files checked: $env_name under $environment_root"

world_model="$source_root/saved_models/$env_name/DeepFM"
for artifact in \
  "$world_model/params/[pointneg]_params.pickle" \
  "$world_model/matsPre/[pointneg]_matPre.pickle" \
  "$world_model/matsVar/[pointneg]_matVar.pickle"; do
  [[ -f "$artifact" ]] || { echo "Missing source world-model artifact: $artifact" >&2; exit 2; }
done
if [[ "$env_name" == KuaiRand-v0 ]]; then
  [[ -f "$world_model/entropy/map_entropy.pickle" ]] || {
    echo "Missing oral entropy map: $world_model/entropy/map_entropy.pickle" >&2; exit 2;
  }
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
cpu_threads="${CPU_THREADS:-3}"
[[ "$cpu_threads" =~ ^[1-3]$ ]] || { echo "CPU_THREADS must be 1..3" >&2; exit 2; }
export OMP_NUM_THREADS="$cpu_threads" MKL_NUM_THREADS="$cpu_threads" OPENBLAS_NUM_THREADS="$cpu_threads"
export NUMEXPR_NUM_THREADS="$cpu_threads" BLIS_NUM_THREADS="$cpu_threads" VECLIB_MAXIMUM_THREADS="$cpu_threads"
export ROLeR_OOD_CPU_THREADS="$cpu_threads" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

default_detector_dir="$output_root/saved_models/$env_name/dorr_shared_ood_seed0"
if [[ "$env_name" == KuaiEnv-v0 ]]; then
  default_detector_dir="$output_root/saved_models/$env_name/dorr_shared_ood"
fi
detector_dir="${DETECTOR_DIR:-$default_detector_dir}"
detector="$detector_dir/detector.pt"
common_roler=(
  --env "$env_name" --cuda 0 --seed 0
  --read_message pointneg --which_tracker "$tracker" --reward_handle cat
  --window_size "$window" --scratch True --num_leave_compute 1 --leave_threshold 0
  --lambda_variance "$variance" --lambda_entropy "$entropy"
  --change_pred_reward True --change_var True --kr "$kr" --ku "$ku"
  --uncertain_type II-weight-norm
)

if [[ "$mode" == prepare || "$mode" == all ]]; then
  if [[ -f "$detector_dir/id_train.npz" && -f "$detector_dir/id_calibration.npz" ]]; then
    echo "Reusing existing ID train/calibration splits: $detector_dir"
    if [[ ! -f "$detector_dir/detector_uncalibrated.pt" && ! -f "$detector" ]]; then
      python run_ood_detector.py train \
        --dataset "$detector_dir/id_train.npz" \
        --checkpoint "$detector_dir/detector_uncalibrated.pt" \
        --architecture shared --task-sampling both --device cuda --seed 0 \
        --epochs 100 --batch-size 256 --learning-rate 3e-4 \
        --hidden-dim 256 --time-embed-dim 16 --num-hidden-layers 4 --score-levels 1
    fi
    if [[ ! -f "$detector" ]]; then
      python run_ood_detector.py calibrate \
        --dataset "$detector_dir/id_calibration.npz" \
        --reference-dataset "$detector_dir/id_train.npz" \
        --checkpoint "$detector_dir/detector_uncalibrated.pt" \
        --output-checkpoint "$detector" \
        --percentiles 90 95 99 --stable-score-noise --device cuda
    fi
  else
    mkdir -p "$detector_dir"
    python run_ood_export.py \
      --source-root "$source_root" --policy-checkpoint "$initial" \
      --output "$detector_dir/id_all.npz" --export-batch-size 1024 \
      "${common_roler[@]}"
    python run_ood_detector.py split \
      --dataset "$detector_dir/id_all.npz" \
      --train-output "$detector_dir/id_train.npz" \
      --calibration-output "$detector_dir/id_calibration.npz" \
      --calibration-fraction 0.2 --seed 0
    python run_ood_detector.py train \
      --dataset "$detector_dir/id_train.npz" \
      --checkpoint "$detector_dir/detector_uncalibrated.pt" \
      --architecture shared --task-sampling both --device cuda --seed 0 \
      --epochs 100 --batch-size 256 --learning-rate 3e-4 \
      --hidden-dim 256 --time-embed-dim 16 --num-hidden-layers 4 --score-levels 1
    python run_ood_detector.py calibrate \
      --dataset "$detector_dir/id_calibration.npz" \
      --reference-dataset "$detector_dir/id_train.npz" \
      --checkpoint "$detector_dir/detector_uncalibrated.pt" \
      --output-checkpoint "$detector" \
      --percentiles 90 95 99 --stable-score-noise --device cuda
  fi
fi

if [[ "$mode" == train || "$mode" == all ]]; then
  for artifact in "$detector" "$detector_dir/id_train.npz" "$detector_dir/id_calibration.npz"; do
    [[ -f "$artifact" ]] || { echo "Missing prepared detector artifact: $artifact" >&2; exit 2; }
  done
  run_name="${RUN_NAME:-DORR_${env_name}_both_seed${seed}}"
  [[ "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || {
    echo "RUN_NAME may contain only letters, digits, underscores, and hyphens." >&2; exit 2;
  }
  run_dir="$output_root/saved_models/$env_name/$run_name"
  [[ ! -e "$run_dir" ]] || { echo "Run already exists: $run_dir" >&2; exit 2; }
  python run_DORR_Main.py \
    --source-root "$source_root" --output-root "$output_root" --run-name "$run_name" \
    --initial-policy-checkpoint "$initial" \
    --action-detector-checkpoint "$detector" \
    --behavior-dataset "$detector_dir/id_train.npz" \
    --detector-calibration-dataset "$detector_dir/id_calibration.npz" \
    --reference-candidates bc --state-ood-policy harmful --q-init-rng isolated \
    --reference-scorer q --lazy-reference --value-comparison reward-bootstrap \
    --disable-offline-q --q-loss-weight 0.05 --q-critic-lr 3e-4 --q-action-embed-dim 64 \
    --harmful-rule legacy --harmful-actor-mode legacy \
    --harmful-decay-start 50 --harmful-decay-end 60 \
    --beta 0 --lambda-beneficial 0 --beneficial-actor-mode guarded \
    --dorr-confidence-source maxvar --device cuda --cpu-threads "$cpu_threads" --seed "$seed" \
    --behavior-candidate-count 10 --action-percentile 95 --state-percentile 95 \
    --margin-ratio 0.05 --actor-harmful-weight 0.30 --actor-beneficial-weight 0.25 \
    --target-tau 0.005 --warmup-epochs 20 --harmful-ramp-epochs 10 \
    --beneficial-warmup-epochs 20 --beneficial-ramp-epochs 10 \
    --save-every 1 --lr 1e-3 --lr-schedule window-cosine --lr-min 1e-4 \
    --lr-decay-start-epoch 30 --lr-decay-end-epoch 100 --q-lr-schedule constant \
    --env "$env_name" --cuda 0 --num_leave_compute 1 --leave_threshold 0 \
    --read_message pointneg --which_tracker "$tracker" --reward_handle cat \
    --window_size "$window" --scratch True \
    --lambda_variance "$variance" --lambda_entropy "$entropy" \
    --change_pred_reward True --change_var True --kr "$kr" --ku "$ku" \
    --uncertain_type II-weight-norm \
    --epoch 200 --step-per-epoch 100000 --episode-per-collect 100 \
    --training-num 100 --test-num 100 --batch-size 1024
  echo "DORR results: $run_dir/dorr_metrics_summary.json"
fi

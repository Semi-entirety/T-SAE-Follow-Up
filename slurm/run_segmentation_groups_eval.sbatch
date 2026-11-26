#!/bin/bash
#SBATCH --job-name=seg_groups
#SBATCH --account=binxuwang_lab
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-20
#SBATCH -o logs/seg_groups_%A_%a.out
#SBATCH -e logs/seg_groups_%A_%a.err
set -euo pipefail

# Seven blocks x three seeds. Override PY and OUT_ROOT when running elsewhere.
BLOCKS=(0 2 4 6 8 10 11)
block=${BLOCKS[$((SLURM_ARRAY_TASK_ID/3))]}
seed=$((SLURM_ARRAY_TASK_ID%3))
STUDY_DIR=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/n/home12/binxuwang/.conda/envs/torch2/bin/python}
OUT_ROOT=${OUT_ROOT:-$STUDY_DIR/results/raw_segmentation_groups}
mkdir -p "$STUDY_DIR/logs" "$OUT_ROOT"

export PYTHONNOUSERSITE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=${TORCH_HOME:-/n/holylabs/kempner_alvarez_lab/Lab/binxuwang/Caches/torch}
export HF_HOME=${HF_HOME:-/n/holylabs/kempner_alvarez_lab/Lab/binxuwang/Caches/huggingface}

"$PY" "$STUDY_DIR/downstream_tasks/segmentation_groups_eval.py" \
  --block "$block" --seed "$seed" --max_seg 500 --max_prop 300 \
  --outdir "$OUT_ROOT/b${block}_s${seed}"

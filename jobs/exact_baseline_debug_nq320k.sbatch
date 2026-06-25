#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_exact_debug_nq
#SBATCH --account=big_data_psaclay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# If this cluster rejects --gres=gpu:h200:1, change only the gres line
# to the local H200 syntax, or fallback to: #SBATCH --gres=gpu:l40s:1

set -euo pipefail

cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("total vram gb:", round(props.total_memory / 1024**3, 2))
PY

python scripts/run_exact_baseline.py \
  --config configs/exact_baseline_nq320k.yaml \
  --overwrite \
  --max-docs 1000 \
  --max-queries 100

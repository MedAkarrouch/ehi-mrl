#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_train_dense
#SBATCH --account=big_data_psaclay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl

module purge
module load python/3.11.2
source /shared/projects/big_data_psaclay/students_M2/melmoussaoui/thesis/venvs/gpt2_cuda_env/bin/activate
source configs/hpc_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=false

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("total vram GB:", torch.cuda.get_device_properties(0).total_memory / 1024**3)
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

python scripts/train_dense_biencoder.py \
  --config configs/train_dense_nq320k_distilbert.yaml

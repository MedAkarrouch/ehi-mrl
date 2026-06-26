#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_dense_bench_ultra
#SBATCH --account=big_data_psaclay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G

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

python scripts/benchmark_dense_batch_size.py \
  --config configs/train_dense_nq320k_distilbert.yaml \
  --batch-sizes 1536,1792,2048,2560 \
  --steps 10 \
  --num-workers 2 \
  --prefetch-factor 2 \
  --persistent-workers false \
  --output results/training_benchmarks/dense_nq320k_h200_batch_sweep_ultramem.csv

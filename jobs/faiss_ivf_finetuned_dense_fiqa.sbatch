#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_faiss_dense_fiqa
#SBATCH --account=big_data_psaclay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=fast
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

set -euo pipefail

cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

python scripts/run_faiss_ivf_sweep.py \
  --config configs/faiss_ivf_finetuned_dense_fiqa.yaml \
  --overwrite

#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_faiss_debug
#SBATCH --account=big_data_psaclay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=fast
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python scripts/run_faiss_ivf_sweep.py \
  --config configs/faiss_ivf_scifact.yaml \
  --overwrite \
  --nlist 32 \
  --nprobe 1,4,8

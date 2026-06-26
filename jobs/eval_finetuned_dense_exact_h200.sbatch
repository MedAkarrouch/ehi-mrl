#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_eval_dense
#SBATCH --account=big_data_psaclay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
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

python scripts/run_finetuned_dense_exact.py \
  --model-dir models/fine_tuned_dense/nq320k_distilbert/final \
  --overwrite

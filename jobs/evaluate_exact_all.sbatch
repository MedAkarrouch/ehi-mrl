#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_eval_exact
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

python scripts/evaluate_run.py --config configs/exact_baseline_nq320k.yaml
python scripts/evaluate_run.py --config configs/exact_baseline_scifact.yaml
python scripts/evaluate_run.py --config configs/exact_baseline_fiqa.yaml

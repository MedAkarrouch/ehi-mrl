#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_validate_data
#SBATCH --account=big_data_psaclay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=fast
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh

python scripts/validate_processed_data.py --config configs/data_nq320k.yaml --strict
python scripts/validate_processed_data.py --config configs/data_beir_scifact.yaml --strict
python scripts/validate_processed_data.py --config configs/data_beir_fiqa.yaml --strict

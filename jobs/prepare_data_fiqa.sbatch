#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_prepare_fiqa
#SBATCH --account=big_data_psaclay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=fast
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail

cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh

python scripts/prepare_data.py --config configs/data_beir_fiqa.yaml --overwrite
python scripts/validate_processed_data.py --config configs/data_beir_fiqa.yaml --strict
python scripts/inspect_dataset.py --config configs/data_beir_fiqa.yaml --processed

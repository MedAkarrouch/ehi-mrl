#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_prepare_data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -euo pipefail
cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh
python scripts/prepare_data.py --config configs/data_nq320k.yaml

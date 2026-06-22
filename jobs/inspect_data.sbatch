#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_inspect_data
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -euo pipefail
cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh
python scripts/inspect_dataset.py --config configs/data_nq320k.yaml

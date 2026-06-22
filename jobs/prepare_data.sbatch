#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_prepare_data
#SBATCH --account=big_data_psaclay

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


#SBATCH --partition=fast

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

set -euo pipefail
cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh
python scripts/prepare_data.py --config configs/data_nq320k.yaml

#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_smoke

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=fast

#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail
cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh
bash testing/run_hpc_tests.sh

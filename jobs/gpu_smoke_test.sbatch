#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_gpu_test
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
##SBATCH --partition=<gpu_partition>
##SBATCH --account=<account>
# The cluster may require a specific H200 hardware profile; adjust the GPU directive if needed.

set -euo pipefail
cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh
which python
python --version
nvidia-smi
python testing/test_gpu.py

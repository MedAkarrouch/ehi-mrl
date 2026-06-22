#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_gpu_test
#SBATCH --account=big_data_psaclay


#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1

#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# The cluster may require a specific H200 hardware profile; adjust the GPU directive if needed.

set -euo pipefail
cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh
which python
python --version
nvidia-smi
python testing/test_gpu.py

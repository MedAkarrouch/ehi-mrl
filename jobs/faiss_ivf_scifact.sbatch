#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_faiss_scifact
#SBATCH --account=big_data_psaclay
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=fast
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd /shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
source configs/hpc_env.sh

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

python - <<'PY'
import faiss
print("faiss ok:", faiss.__version__ if hasattr(faiss, "__version__") else "installed")
print("faiss omp threads before:", faiss.omp_get_max_threads())
PY

python scripts/run_faiss_ivf_sweep.py \
  --config configs/faiss_ivf_scifact.yaml \
  --overwrite

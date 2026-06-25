#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_plot_faiss
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

python - <<'PY'
import matplotlib
print("matplotlib:", matplotlib.__version__)
print("backend:", matplotlib.get_backend())
PY

python scripts/plot_faiss_ivf_sweeps.py --overwrite

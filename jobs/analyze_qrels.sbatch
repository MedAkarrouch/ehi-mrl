#!/usr/bin/env bash
#SBATCH --job-name=ehi_mrl_analyze_qrels
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

python scripts/analyze_qrels.py \
  --config configs/data_nq320k.yaml \
  --output-json data/processed/nq320k/qrels_analysis.json \
  --output-csv data/processed/nq320k/qrels_analysis.csv

python scripts/analyze_qrels.py \
  --config configs/data_beir_scifact.yaml \
  --output-json data/processed/beir_scifact/qrels_analysis.json \
  --output-csv data/processed/beir_scifact/qrels_analysis.csv

python scripts/analyze_qrels.py \
  --config configs/data_beir_fiqa.yaml \
  --output-json data/processed/beir_fiqa/qrels_analysis.json \
  --output-csv data/processed/beir_fiqa/qrels_analysis.csv

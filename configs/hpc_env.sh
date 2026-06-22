#!/usr/bin/env bash

module purge
module load python/3.11.2

export PROJECT_ROOT="/shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl"
export VENV_PATH="/shared/projects/big_data_psaclay/students_M2/melmoussaoui/thesis/venvs/gpt2_cuda_env"

source "$VENV_PATH/bin/activate"

export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/.cache/huggingface/transformers"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$PROJECT_ROOT/logs"

# SLURM jobs

Submit all SLURM jobs from the repository root. The expected HPC checkout path is:

```text
/shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
```

Each job sources `configs/hpc_env.sh`, which activates the existing project environment and configures the Hugging Face cache at `.cache/huggingface`. Job stdout and stderr are written to `logs/`.

## Phase 1 data jobs

Phase 1 data preparation, inspection, and validation are CPU-only. These jobs use `partition=fast`; do not request a GPU for data processing.

Prepare the three configured datasets:

```bash
sbatch jobs/prepare_data_nq320k.sbatch
sbatch jobs/prepare_data_scifact.sbatch
sbatch jobs/prepare_data_fiqa.sbatch
```

Validate the normalized outputs after preparation:

```bash
sbatch jobs/validate_all_processed_data.sbatch
```

The GPU template remains reserved for later embedding-generation and training phases.

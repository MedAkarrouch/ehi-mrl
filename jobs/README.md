# SLURM jobs

Submit all SLURM jobs from the repository root. The expected HPC checkout path is:

```text
/shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
```

Each job sources `configs/hpc_env.sh`, which activates the existing project environment and configures the Hugging Face cache at `.cache/huggingface`. Job stdout and stderr are written to `logs/`.

Partition and account directives are intentionally left as comments because the cluster-specific values are not yet known.

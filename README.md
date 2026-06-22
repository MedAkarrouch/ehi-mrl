# EHI-MRL: End-to-End Hierarchical Indexing with Matryoshka Representations

EHI-MRL is a research project exploring efficient dense retrieval through end-to-end hierarchical indexing and Matryoshka Representation Learning. This repository currently contains only the Phase 0 project skeleton, dataset configuration scaffolding, HPC job templates, and validation scripts.

## Planned workflow

1. Phase 0: repository skeleton + data scaffolding
2. Phase 1: unified data processing
3. Phase 2: exact-search baseline
4. Phase 3: FAISS IVF baseline
5. Phase 4: dense fine-tuning
6. Phase 5: MRL-only baseline
7. Phase 6: rigid EHI
8. Phase 7: EHI-MRL
9. Phase 8: experiments and plots
10. Phase 9: HPC scaling

## Dataset plan

- **NQ320K**: training + in-domain evaluation
- **BEIR SciFact**: out-of-distribution evaluation
- **BEIR FiQA**: out-of-distribution evaluation

Datasets are loaded from the Hugging Face Hub only when the data scripts run. They are cached locally or on HPC and are not stored in Git.

## HPC environment

- Project path: `/shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl`
- Existing environment: `/shared/projects/big_data_psaclay/students_M2/melmoussaoui/thesis/venvs/gpt2_cuda_env`

The project uses the existing CUDA-compatible environment. Do not create a new virtual environment or install/reinstall PyTorch.

## Basic usage

Run the local, Codex-safe checks:

```bash
bash testing/run_codex_tests.sh
```

On HPC, install missing project dependencies into the existing environment:

```bash
bash scripts/install_project_deps.sh
```

Then run the HPC checks:

```bash
bash testing/run_hpc_tests.sh
```

Submit the smoke-test jobs from the repository root:

```bash
sbatch jobs/smoke_test.sbatch
sbatch jobs/gpu_smoke_test.sbatch
```

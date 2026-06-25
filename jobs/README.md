# SLURM jobs

Submit all SLURM jobs from the repository root. The expected HPC checkout path is:

```text
/shared/projects/big_data_psaclay/students_M2/melmoussaoui/ehi-mrl
```

Each job sources `configs/hpc_env.sh`, which activates the existing project environment and configures the Hugging Face cache at `.cache/huggingface`. Job stdout and stderr are written to `logs/`.

## Phase 1 data jobs

Phase 1 data preparation, inspection, and validation are CPU-only. These jobs use `partition=fast`; do not request a GPU for data processing.

BEIR datasets expose separate Hugging Face configurations for `corpus` and `queries`. Pull this fix on HPC before preparing SciFact or FiQA.

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

Qrels analysis:

```bash
sbatch jobs/analyze_qrels.sbatch
```

This is a CPU-only diagnostic job. It reports how many relevant documents each query has in each split.

## Phase 2: exact dense baseline

These jobs use the frozen encoder:
sentence-transformers/distilbert-base-nli-stsb-mean-tokens

They create embeddings, run exact cosine search, and compute retrieval metrics.

The full GPU jobs prefer H200 because it has much larger VRAM than L40S and reduces out-of-memory risk during exact dense search. If the cluster rejects `--gres=gpu:h200:1`, change only the gres line to the local H200 syntax or fallback to `--gres=gpu:l40s:1`.

Debug:

```bash
sbatch jobs/exact_baseline_debug_nq320k.sbatch
```

Full runs:

```bash
sbatch jobs/exact_baseline_nq320k.sbatch
sbatch jobs/exact_baseline_scifact.sbatch
sbatch jobs/exact_baseline_fiqa.sbatch
```

Metrics-only rerun:

```bash
sbatch jobs/evaluate_exact_all.sbatch
```

## Phase 3: FAISS IVF baseline

Phase 3 reuses Phase 2 frozen SBERT embeddings and builds post-hoc FAISS IVF indexes. This is CPU-only and uses `faiss-cpu`; do not request GPU and do not install FAISS-GPU.

Debug:

```bash
sbatch jobs/faiss_ivf_debug_scifact.sbatch
```

Full sweeps:

```bash
sbatch jobs/faiss_ivf_nq320k.sbatch
sbatch jobs/faiss_ivf_scifact.sbatch
sbatch jobs/faiss_ivf_fiqa.sbatch
```

Outputs:

```text
data/indexes/faiss_ivf/{dataset}/sbert_distilbert_nli_stsb/
results/faiss_ivf/{dataset}/sbert_distilbert_nli_stsb/sweep_summary.csv
```

## Phase 3b: FAISS IVF plots

Phase 3b reads Phase 2 exact metrics and Phase 3 FAISS IVF sweep summaries, then generates publication-quality Matplotlib figures.

The plotting script saves every figure as both SVG and PNG:

- SVG for high-quality vector output
- PNG for quick viewing and slides

Run:

```bash
sbatch jobs/plot_faiss_ivf.sbatch
```

Optional log-scale x-axis version:

```bash
sbatch jobs/plot_faiss_ivf_logx.sbatch
```

Outputs:

```text
results/plots/faiss_ivf/
results/plots/faiss_ivf_logx/
```

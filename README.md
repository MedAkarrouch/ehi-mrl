# EHI-MRL: End-to-End Hierarchical Indexing with Matryoshka Representations

EHI-MRL is a research project exploring efficient dense retrieval through end-to-end hierarchical indexing and Matryoshka Representation Learning. Phase 1 provides unified data normalization and validation; model training, indexing, embeddings, and retrieval evaluation are intentionally deferred.

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

## Phase 1: normalized retrieval data

Processed files are written under `data/processed/{dataset_name}/` using BEIR-compatible conventions:

```text
corpus.jsonl
queries_train.jsonl | queries_dev.jsonl | queries_test.jsonl
qrels_train.tsv | qrels_dev.tsv | qrels_test.tsv
triples_train.tsv
dataset_info.json
```

Corpus rows use `{"_id": "doc-id", "title": "", "text": "document text"}` and query rows use `{"_id": "query-id", "text": "query text"}`. Qrels and training triples are tab-separated files with explicit headers.

NQ320K is normalized for training and in-domain evaluation, including one deterministic sampled negative per training query. SciFact and FiQA are normalized for out-of-distribution evaluation only, without default training triples; their Hugging Face sources use separate `corpus` and `queries` configs. Source datasets are cached from Hugging Face and their normalized outputs remain ignored by Git.

### Phase 1 diagnostic

`scripts/analyze_qrels.py` reports qrels statistics, including the distribution of relevant documents per query. This helps verify whether a dataset behaves like a pair-based dataset, such as NQ320K, or a qrels-based evaluation dataset with potentially multiple relevant documents per query, such as BEIR SciFact and BEIR FiQA.

Run the offline, Codex-safe tests (they do not download datasets):

```bash
bash testing/run_codex_tests.sh
```

On HPC, Phase 1 data jobs are CPU-only and use the `fast` partition:

```bash
sbatch jobs/prepare_data_nq320k.sbatch
sbatch jobs/prepare_data_scifact.sbatch
sbatch jobs/prepare_data_fiqa.sbatch
sbatch jobs/validate_all_processed_data.sbatch
```

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

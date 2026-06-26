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

## Phase 2: frozen exact dense retrieval baseline

Phase 2 establishes a frozen imported encoder baseline before MRL, FAISS, and EHI. It is not training and does not fine-tune the model.

- Encoder: `sentence-transformers/distilbert-base-nli-stsb-mean-tokens`
- Similarity: cosine, implemented as dot product over normalized `float32` embeddings
- Search: exact dense retrieval, comparing every query embedding against every corpus embedding in query batches and corpus chunks
- Main jobs: create embeddings, write exact run files, and compute retrieval metrics

Primary metrics:

- NQ320K dev: Hit@1, MRR@10, Recall@10, Recall@100
- BEIR SciFact test: nDCG@10, Recall@100, MRR@10, Hit@1
- BEIR FiQA test: nDCG@10, Recall@100, MRR@10, Hit@1

Evaluation averages only over qrels-covered queries. `Hit@1` means relevance-based top-1 correctness. Do not call this `NN@1`; `NN@1` is reserved for later exact-vs-approximate index agreement in FAISS/EHI phases.

Full embedding/search jobs prefer H200 to reduce exact-search out-of-memory risk. L40S is documented only as a fallback in the Slurm job comments.

## Phase 3: FAISS IVF approximate-search baseline

Phase 3 is a post-hoc FAISS IVF approximate-search baseline. It reuses the frozen SBERT embeddings created by Phase 2, does not train or fine-tune the encoder, and does not use GPU FAISS.

This phase uses `faiss-cpu` to build IVF indexes for different `nlist` values and search them with different `nprobe` values. FAISS uses inner product because Phase 2 embeddings are already normalized, so inner product is cosine similarity.

The sweep reports the same relevance metrics as Phase 2 plus efficiency diagnostics:

- `%DocsVisited`
- `AvgDocsVisited`
- `LatencyMsPerQuery`

This measures the standard quality-efficiency trade-off of disjoint ANNS indexing. It is the conventional post-hoc baseline that later EHI and EHI-MRL phases must beat.

## Phase 3b: FAISS IVF result aggregation and plotting

Phase 3b does not run retrieval. It reads existing Phase 2 exact metric JSON files and Phase 3 FAISS IVF `sweep_summary.csv` files, then generates quality-vs-efficiency figures.

The main x-axis is `% documents visited`. Each figure is saved as both SVG for high-quality vector output and PNG for quick viewing or slides. The plotting script also writes `best_operating_points.csv`.

Main plots:

- NQ320K: Recall@100, Recall@10, MRR@10
- BEIR SciFact: nDCG@10, Recall@100, MRR@10
- BEIR FiQA: nDCG@10, Recall@100, MRR@10

## Phase 4: Fine-tuned Dense Retriever

This phase trains a rigid dense dual encoder on NQ320K using in-batch contrastive learning. It produces 768-dimensional query/document embeddings and evaluates both exhaustive exact search and post-hoc FAISS-IVF search.

Method labels:

- Fine-tuned Dense + Exact Search
- Fine-tuned Dense + FAISS-IVF

This phase does not use MRL, EHI, learned routing, or adaptive dimensions.

Phase 4 does not calculate or report Hit@1.

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

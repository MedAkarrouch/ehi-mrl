"""Helpers for CPU FAISS IVF baselines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from retrieval_utils import load_id_list


def import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss is required for Phase 3. Install project dependencies with faiss-cpu.") from exc
    return faiss


def import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for FAISS embedding I/O. Install project dependencies first.") from exc
    return np


def faiss_version(faiss: Any) -> str:
    return str(faiss.__version__) if hasattr(faiss, "__version__") else "installed"


def resolve_omp_threads(config_value: Any) -> int | None:
    if config_value is None or str(config_value).lower() == "auto":
        env_value = os.environ.get("OMP_NUM_THREADS")
        return int(env_value) if env_value and env_value.isdigit() else None
    return int(config_value)


def set_faiss_threads(faiss: Any, config_value: Any) -> int:
    requested_threads = resolve_omp_threads(config_value)
    if requested_threads is not None and requested_threads > 0:
        faiss.omp_set_num_threads(requested_threads)
    return int(faiss.omp_get_max_threads())


def load_float32_embeddings(path: str | Path) -> Any:
    np = import_numpy()
    source = Path(path)
    embeddings = np.load(source)
    if embeddings.ndim != 2:
        raise RuntimeError(f"{source.name} must be a 2D embedding array; found shape {embeddings.shape}.")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    return np.ascontiguousarray(embeddings)


def validate_embedding_ids(embeddings: Any, ids: list[str], name: str) -> None:
    if embeddings.ndim != 2:
        raise RuntimeError(f"{name} embeddings must be 2D.")
    if embeddings.shape[0] != len(ids):
        raise RuntimeError(f"{name} embedding count {embeddings.shape[0]} does not match ID count {len(ids)}.")
    if embeddings.shape[0] == 0:
        raise RuntimeError(f"{name} embeddings are empty.")


def load_embeddings_and_ids(embedding_path: str | Path, id_path: str | Path, name: str) -> tuple[Any, list[str]]:
    embeddings = load_float32_embeddings(embedding_path)
    ids = load_id_list(id_path)
    validate_embedding_ids(embeddings, ids, name)
    return embeddings, ids


def training_subset(corpus_embeddings: Any, max_train_vectors: int | None) -> Any:
    if max_train_vectors is None or max_train_vectors >= corpus_embeddings.shape[0]:
        return corpus_embeddings
    if max_train_vectors <= 0:
        raise RuntimeError("max_train_vectors must be positive when set.")
    np = import_numpy()
    indices = np.linspace(0, corpus_embeddings.shape[0] - 1, num=max_train_vectors, dtype=np.int64)
    return corpus_embeddings[indices]


def build_ivf_flat_ip_index(
    faiss: Any,
    corpus_embeddings: Any,
    nlist: int,
    max_train_vectors: int | None = None,
) -> tuple[Any, int]:
    if nlist <= 0:
        raise RuntimeError("nlist must be positive.")
    corpus_size, dim = corpus_embeddings.shape
    if nlist > corpus_size:
        raise RuntimeError(f"nlist={nlist} is larger than corpus size {corpus_size}.")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    train_vectors = training_subset(corpus_embeddings, max_train_vectors)
    index.train(train_vectors)
    index.add(corpus_embeddings)
    return index, int(train_vectors.shape[0])


def save_index(faiss: Any, index: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(target))


def load_index(faiss: Any, path: str | Path) -> Any:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"FAISS index does not exist: {source}")
    return faiss.read_index(str(source))


def compute_probed_list_ids(index: Any, query_embeddings: Any, nprobe: int) -> Any:
    if nprobe <= 0:
        raise RuntimeError("nprobe must be positive.")
    _scores, list_ids = index.quantizer.search(query_embeddings, nprobe)
    return list_ids


def compute_avg_docs_visited(index: Any, query_embeddings: Any, nprobe: int) -> float:
    if query_embeddings.shape[0] == 0:
        return 0.0
    list_ids = compute_probed_list_ids(index, query_embeddings, nprobe)
    total = 0
    for row in list_ids:
        for list_id in row:
            if int(list_id) >= 0:
                total += int(index.invlists.list_size(int(list_id)))
    return total / query_embeddings.shape[0]

#!/usr/bin/env python3
"""Build a CPU FAISS IndexIVFFlat over Phase 2 corpus embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from data_utils import ensure_dir
from faiss_utils import (
    build_ivf_flat_ip_index,
    faiss_version,
    import_faiss,
    load_embeddings_and_ids,
    save_index,
    set_faiss_threads,
)
from retrieval_utils import load_config, repo_root_from_script, resolve_path


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a FAISS IVF YAML config.")
    parser.add_argument("--nlist", required=True, type=int, help="Number of IVF lists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index outputs.")
    args = parser.parse_args()

    try:
        repo_root = repo_root_from_script(__file__)
        config = load_config(resolve_path(repo_root, args.config))
        if config.get("metric") != "inner_product":
            raise RuntimeError("SBERT + FAISS-IVF currently supports only metric: inner_product.")

        index_dir = resolve_path(repo_root, config["index_dir"])
        embedding_dir = resolve_path(repo_root, config["embedding_dir"])
        index_path = index_dir / f"ivf_nlist{args.nlist}.faiss"
        info_path = index_dir / f"ivf_nlist{args.nlist}_info.json"
        if not args.overwrite:
            existing = [path for path in (index_path, info_path) if path.exists()]
            if existing:
                names = ", ".join(str(path) for path in existing)
                raise RuntimeError(f"Refusing to overwrite existing FAISS output(s): {names}. Use --overwrite.")

        faiss = import_faiss()
        omp_threads = set_faiss_threads(faiss, config.get("omp_threads", "auto"))
        corpus_embeddings, corpus_ids = load_embeddings_and_ids(
            embedding_dir / "corpus_embeddings.npy", embedding_dir / "corpus_ids.json", "corpus"
        )
        max_train_vectors = config.get("max_train_vectors")
        if max_train_vectors is not None:
            max_train_vectors = int(max_train_vectors)

        print(f"Building FAISS IVF index: dataset={config['dataset_name']} nlist={args.nlist}")
        print(f"corpus vectors: {corpus_embeddings.shape[0]}, dim: {corpus_embeddings.shape[1]}")
        index, train_count = build_ivf_flat_ip_index(faiss, corpus_embeddings, args.nlist, max_train_vectors)
        save_index(faiss, index, index_path)
        info = {
            "dataset_name": config["dataset_name"],
            "embedding_dir": str(embedding_dir),
            "index_path": str(index_path),
            "corpus_count": len(corpus_ids),
            "embedding_dimension": int(corpus_embeddings.shape[1]),
            "nlist": args.nlist,
            "metric": config["metric"],
            "max_train_vectors": max_train_vectors,
            "actual_training_vector_count": train_count,
            "faiss_version": faiss_version(faiss),
            "omp_threads": omp_threads,
        }
        write_json(info_path, info)
        print(f"Wrote FAISS index: {index_path}")
        print(f"Wrote index info: {info_path}")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

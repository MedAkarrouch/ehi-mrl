#!/usr/bin/env python3
"""Search a saved CPU FAISS IVF index and write a Phase 2-compatible run file."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from data_utils import ensure_dir
from faiss_utils import (
    compute_avg_docs_visited,
    import_faiss,
    load_embeddings_and_ids,
    load_index,
    set_faiss_threads,
)
from retrieval_utils import load_config, load_id_list, repo_root_from_script, resolve_path, write_run_tsv


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a FAISS IVF YAML config.")
    parser.add_argument("--nlist", required=True, type=int, help="Number of IVF lists used by the saved index.")
    parser.add_argument("--nprobe", required=True, type=int, help="Number of IVF lists to probe.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run/search outputs.")
    args = parser.parse_args()

    try:
        repo_root = repo_root_from_script(__file__)
        config = load_config(resolve_path(repo_root, args.config))
        exact_config = load_config(resolve_path(repo_root, config["exact_baseline_config"]))
        split = str(config["split"])
        if args.nprobe <= 0:
            raise RuntimeError("nprobe must be positive.")
        if args.nprobe > args.nlist:
            raise RuntimeError(f"nprobe={args.nprobe} is larger than nlist={args.nlist}.")

        embedding_dir = resolve_path(repo_root, config["embedding_dir"])
        index_dir = resolve_path(repo_root, config["index_dir"])
        results_dir = resolve_path(repo_root, config["results_dir"])
        run_path = results_dir / f"run_{split}_nlist{args.nlist}_nprobe{args.nprobe}.tsv"
        info_path = results_dir / f"search_info_{split}_nlist{args.nlist}_nprobe{args.nprobe}.json"
        if not args.overwrite:
            existing = [path for path in (run_path, info_path) if path.exists()]
            if existing:
                names = ", ".join(str(path) for path in existing)
                raise RuntimeError(f"Refusing to overwrite existing FAISS search output(s): {names}. Use --overwrite.")

        faiss = import_faiss()
        omp_threads = set_faiss_threads(faiss, config.get("omp_threads", "auto"))
        index = load_index(faiss, index_dir / f"ivf_nlist{args.nlist}.faiss")
        query_embeddings, query_ids = load_embeddings_and_ids(
            embedding_dir / f"queries_{split}_embeddings.npy",
            embedding_dir / f"queries_{split}_ids.json",
            "query",
        )
        corpus_ids = load_id_list(embedding_dir / "corpus_ids.json")
        if index.d != query_embeddings.shape[1]:
            raise RuntimeError(f"Index dimension {index.d} does not match query dimension {query_embeddings.shape[1]}.")
        if index.ntotal != len(corpus_ids):
            raise RuntimeError(f"Index vector count {index.ntotal} does not match corpus ID count {len(corpus_ids)}.")

        top_k = min(int(config.get("top_k", exact_config.get("top_k", 100))), len(corpus_ids))
        index.nprobe = args.nprobe
        print(
            f"Searching FAISS IVF: dataset={config['dataset_name']} split={split} "
            f"nlist={args.nlist} nprobe={args.nprobe} top_k={top_k}"
        )
        start = time.perf_counter()
        scores, indices = index.search(query_embeddings, top_k)
        search_seconds = time.perf_counter() - start
        rankings: dict[str, list[tuple[str, float]]] = {}
        for row_index, query_id in enumerate(query_ids):
            docs: list[tuple[str, float]] = []
            for corpus_index, score in zip(indices[row_index], scores[row_index]):
                if int(corpus_index) >= 0:
                    docs.append((corpus_ids[int(corpus_index)], float(score)))
            rankings[query_id] = docs
        write_run_tsv(run_path, rankings)

        avg_docs_visited = compute_avg_docs_visited(index, query_embeddings, args.nprobe)
        percent_docs_visited = 100.0 * avg_docs_visited / len(corpus_ids) if corpus_ids else 0.0
        latency_ms = 1000.0 * search_seconds / len(query_ids) if query_ids else 0.0
        info = {
            "dataset_name": config["dataset_name"],
            "split": split,
            "nlist": args.nlist,
            "nprobe": args.nprobe,
            "query_count": len(query_ids),
            "corpus_count": len(corpus_ids),
            "top_k": top_k,
            "metric": config["metric"],
            "search_seconds": search_seconds,
            "latency_ms_per_query": latency_ms,
            "avg_docs_visited": avg_docs_visited,
            "percent_docs_visited": percent_docs_visited,
            "omp_threads": omp_threads,
        }
        write_json(info_path, info)
        print(f"Wrote FAISS run: {run_path}")
        print(f"LatencyMsPerQuery: {latency_ms:.4f}")
        print(f"AvgDocsVisited: {avg_docs_visited:.2f}")
        print(f"%DocsVisited: {percent_docs_visited:.4f}")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

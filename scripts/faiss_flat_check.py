#!/usr/bin/env python3
"""Check FAISS IndexFlatIP agreement with the Phase 2 exact run file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from faiss_utils import import_faiss, load_embeddings_and_ids
from retrieval_utils import load_config, load_run_tsv, repo_root_from_script, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-config", required=True, type=Path, help="Path to a Phase 2 exact baseline config.")
    parser.add_argument("--max-queries", type=int, default=100, help="Maximum queries to check.")
    args = parser.parse_args()

    try:
        repo_root = repo_root_from_script(__file__)
        config = load_config(resolve_path(repo_root, args.exact_config))
        split = str(config["split"])
        embedding_dir = resolve_path(repo_root, config["embedding_dir"])
        results_dir = resolve_path(repo_root, config["results_dir"])
        exact_run_path = results_dir / f"run_{split}.tsv"
        if not exact_run_path.is_file():
            raise RuntimeError(f"Phase 2 exact run file does not exist: {exact_run_path}")

        faiss = import_faiss()
        corpus_embeddings, corpus_ids = load_embeddings_and_ids(
            embedding_dir / "corpus_embeddings.npy", embedding_dir / "corpus_ids.json", "corpus"
        )
        query_embeddings, query_ids = load_embeddings_and_ids(
            embedding_dir / f"queries_{split}_embeddings.npy",
            embedding_dir / f"queries_{split}_ids.json",
            "query",
        )
        checked_query_ids = query_ids[: args.max_queries]
        checked_embeddings = query_embeddings[: len(checked_query_ids)]
        top_k = min(int(config.get("top_k", 100)), 10, len(corpus_ids))

        index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        index.add(corpus_embeddings)
        _scores, indices = index.search(checked_embeddings, top_k)
        faiss_rankings = {
            query_id: [corpus_ids[int(index_value)] for index_value in row if int(index_value) >= 0]
            for query_id, row in zip(checked_query_ids, indices)
        }
        exact_run = load_run_tsv(exact_run_path)
        comparable_query_ids = [query_id for query_id in checked_query_ids if query_id in exact_run]
        if not comparable_query_ids:
            raise RuntimeError("No checked queries were present in the Phase 2 exact run file.")

        top1_matches = 0
        top10_overlap_total = 0.0
        for query_id in comparable_query_ids:
            exact_docs = list(exact_run[query_id].keys())[:10]
            faiss_docs = faiss_rankings[query_id][:10]
            if exact_docs[:1] and faiss_docs[:1] and exact_docs[0] == faiss_docs[0]:
                top1_matches += 1
            denominator = min(10, len(exact_docs), len(faiss_docs)) or 1
            top10_overlap_total += len(set(exact_docs) & set(faiss_docs)) / denominator

        top1_agreement = top1_matches / len(comparable_query_ids)
        average_top10_overlap = top10_overlap_total / len(comparable_query_ids)
        print(f"checked query count: {len(comparable_query_ids)}")
        print(f"top1 agreement: {top1_agreement:.6f}")
        print(f"average top10 overlap: {average_top10_overlap:.6f}")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

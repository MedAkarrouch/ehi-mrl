#!/usr/bin/env python3
"""Embed a processed retrieval dataset with a frozen SentenceTransformer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

from data_utils import ensure_dir
from retrieval_utils import (
    load_config,
    load_corpus_jsonl,
    load_id_list,
    load_queries_jsonl,
    repo_root_from_script,
    resolve_path,
    save_id_list,
)


def import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def select_device(requested_device: str) -> tuple[str, Any | None]:
    torch = import_torch()
    requested = requested_device.lower()
    if requested == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda", torch
        return "cpu", torch
    if requested.startswith("cuda"):
        if torch is None:
            raise RuntimeError("CUDA was requested, but torch is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return requested_device, torch
    if requested == "cpu":
        return "cpu", torch
    raise RuntimeError(f"Unsupported device setting: {requested_device}")


def cuda_info(torch: Any | None, device: str) -> dict[str, Any]:
    if torch is None or not device.startswith("cuda") or not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "total_gpu_memory_gb": round(props.total_memory / 1024**3, 2),
    }


def print_cuda_startup(torch: Any | None, device: str) -> None:
    info = cuda_info(torch, device)
    if info:
        print(f"Using CUDA device: {info['gpu_name']}")
        print(f"total vram gb: {info['total_gpu_memory_gb']}")
        torch.cuda.reset_peak_memory_stats()


def encode_texts(
    model: Any,
    texts: list[str],
    batch_size: int,
    normalize_embeddings: bool,
    dtype: str,
    kind: str,
) -> np.ndarray:
    if np is None:
        raise RuntimeError("numpy is required for embedding output. Install project dependencies first.")
    method_name = "encode_document" if kind == "document" else "encode_query"
    method = getattr(model, method_name, None)
    if method is None:
        method = model.encode
    try:
        embeddings = method(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
    except TypeError:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
    return np.asarray(embeddings, dtype=dtype)


def check_outputs_do_not_exist(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        names = ", ".join(str(path) for path in existing)
        raise RuntimeError(f"Refusing to overwrite existing embedding output(s): {names}. Use --overwrite.")


def existing_id_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    return len(load_id_list(path))


def write_info(path: Path, info: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to an exact baseline YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing embedding outputs.")
    parser.add_argument("--encode-corpus", action="store_true", help="Encode the corpus only, unless --encode-queries is also set.")
    parser.add_argument("--encode-queries", action="store_true", help="Encode the configured query split only.")
    parser.add_argument("--max-docs", type=int, help="Optional debug limit for corpus rows.")
    parser.add_argument("--max-queries", type=int, help="Optional debug limit for query rows.")
    args = parser.parse_args()

    try:
        repo_root = repo_root_from_script(__file__)
        if np is None:
            raise RuntimeError("numpy is required for embedding output. Install project dependencies first.")
        baseline_config = load_config(resolve_path(repo_root, args.config))
        dataset_config = load_config(resolve_path(repo_root, baseline_config["dataset_config"]))
        encoder_config = load_config(resolve_path(repo_root, baseline_config["encoder_config"]))

        encode_corpus = args.encode_corpus or not args.encode_queries
        encode_queries = args.encode_queries or not args.encode_corpus

        processed_dir = resolve_path(repo_root, dataset_config["output_dir"])
        query_path = processed_dir / str(baseline_config["query_file"])
        corpus_path = processed_dir / "corpus.jsonl"
        embedding_dir = resolve_path(repo_root, baseline_config["embedding_dir"])
        split = str(baseline_config["split"])

        corpus_embeddings_path = embedding_dir / "corpus_embeddings.npy"
        corpus_ids_path = embedding_dir / "corpus_ids.json"
        query_embeddings_path = embedding_dir / f"queries_{split}_embeddings.npy"
        query_ids_path = embedding_dir / f"queries_{split}_ids.json"
        info_path = embedding_dir / "embedding_info.json"

        output_paths: list[Path] = [info_path]
        if encode_corpus:
            output_paths.extend([corpus_embeddings_path, corpus_ids_path])
        if encode_queries:
            output_paths.extend([query_embeddings_path, query_ids_path])
        check_outputs_do_not_exist(output_paths, args.overwrite)

        requested_device = str(baseline_config.get("device") or encoder_config.get("device") or "auto")
        device, torch = select_device(requested_device)
        print_cuda_startup(torch, device)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is required for embedding. Install project dependencies first.") from exc

        model_name = str(encoder_config["hf_model_name"])
        model = SentenceTransformer(model_name, device=device)
        max_seq_length = encoder_config.get("max_seq_length")
        if max_seq_length is not None and hasattr(model, "max_seq_length"):
            model.max_seq_length = int(max_seq_length)

        dtype = str(encoder_config.get("embedding_dtype", "float32"))
        normalize_embeddings = bool(encoder_config.get("normalize_embeddings", True))
        document_batch_size = int(encoder_config.get("document_batch_size", 256))
        query_batch_size = int(encoder_config.get("query_batch_size", 512))

        corpus_count: int | None = existing_id_count(corpus_ids_path)
        query_count: int | None = existing_id_count(query_ids_path)
        embedding_dim: int | None = None

        ensure_dir(embedding_dir)
        if encode_corpus:
            corpus_rows = load_corpus_jsonl(corpus_path, max_docs=args.max_docs)
            corpus_ids = [doc_id for doc_id, _text in corpus_rows]
            corpus_texts = [text for _doc_id, text in corpus_rows]
            print(f"Encoding corpus rows: {len(corpus_rows)}")
            corpus_embeddings = encode_texts(
                model, corpus_texts, document_batch_size, normalize_embeddings, dtype, kind="document"
            )
            np.save(corpus_embeddings_path, corpus_embeddings)
            save_id_list(corpus_ids_path, corpus_ids)
            corpus_count = len(corpus_ids)
            embedding_dim = int(corpus_embeddings.shape[1]) if corpus_embeddings.ndim == 2 else None

        if encode_queries:
            query_rows = load_queries_jsonl(query_path, max_queries=args.max_queries)
            query_ids = [query_id for query_id, _text in query_rows]
            query_texts = [text for _query_id, text in query_rows]
            print(f"Encoding {split} query rows: {len(query_rows)}")
            query_embeddings = encode_texts(
                model, query_texts, query_batch_size, normalize_embeddings, dtype, kind="query"
            )
            np.save(query_embeddings_path, query_embeddings)
            save_id_list(query_ids_path, query_ids)
            query_count = len(query_ids)
            embedding_dim = embedding_dim or (int(query_embeddings.shape[1]) if query_embeddings.ndim == 2 else None)

        info: dict[str, Any] = {
            "encoder_name": encoder_config["encoder_name"],
            "hf_model_name": model_name,
            "dataset_name": baseline_config["dataset_name"],
            "split": split,
            "corpus_count": corpus_count,
            "query_count": query_count,
            "embedding_dimension": embedding_dim,
            "normalized": normalize_embeddings,
            "dtype": dtype,
            "device_used": device,
            "max_docs": args.max_docs,
            "max_queries": args.max_queries,
        }
        info.update(cuda_info(torch, device))
        if torch is not None and device.startswith("cuda") and torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            print("peak cuda memory allocated GB:", peak)
            info["peak_cuda_memory_allocated_gb"] = peak
        write_info(info_path, info)
        print(f"Wrote embeddings to: {embedding_dir}")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

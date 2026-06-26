#!/usr/bin/env python3
"""Encode processed retrieval data with a fine-tuned dense Hugging Face model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from data_utils import ensure_dir
from retrieval_utils import load_config, load_corpus_jsonl, load_queries_jsonl, repo_root_from_script, resolve_path, save_id_list


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def check_outputs(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        raise RuntimeError(f"Refusing to overwrite existing embedding outputs: {', '.join(str(path) for path in existing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from dense_modeling import encode_texts, load_dense_model

        repo_root = repo_root_from_script(__file__)
        dataset_config = load_config(resolve_path(repo_root, args.dataset_config))
        processed_dir = resolve_path(repo_root, dataset_config["output_dir"])
        output_dir = resolve_path(repo_root, args.output_dir)
        corpus_embedding_path = output_dir / "corpus_embeddings.npy"
        corpus_ids_path = output_dir / "corpus_ids.json"
        query_embedding_path = output_dir / f"queries_{args.split}_embeddings.npy"
        query_ids_path = output_dir / f"queries_{args.split}_ids.json"
        info_path = output_dir / "embedding_info.json"
        check_outputs([corpus_embedding_path, corpus_ids_path, query_embedding_path, query_ids_path, info_path], args.overwrite)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer, model = load_dense_model(resolve_path(repo_root, args.model_dir))
        model.to(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            print("gpu:", torch.cuda.get_device_name(0))

        corpus = load_corpus_jsonl(processed_dir / "corpus.jsonl")
        queries = load_queries_jsonl(processed_dir / f"queries_{args.split}.jsonl")
        corpus_ids = [doc_id for doc_id, _text in corpus]
        query_ids = [query_id for query_id, _text in queries]
        ensure_dir(output_dir)
        corpus_embeddings = encode_texts(model, tokenizer, [text for _doc_id, text in corpus], args.batch_size, args.max_length, device, use_bf16=True)
        query_embeddings = encode_texts(model, tokenizer, [text for _query_id, text in queries], args.batch_size, args.max_length, device, use_bf16=True)
        np.save(corpus_embedding_path, corpus_embeddings.numpy().astype("float32"))
        np.save(query_embedding_path, query_embeddings.numpy().astype("float32"))
        save_id_list(corpus_ids_path, corpus_ids)
        save_id_list(query_ids_path, query_ids)
        info = {
            "model_dir": str(resolve_path(repo_root, args.model_dir)),
            "dataset_name": dataset_config["dataset_name"],
            "split": args.split,
            "corpus_count": len(corpus_ids),
            "query_count": len(query_ids),
            "embedding_dimension": int(corpus_embeddings.shape[1]) if corpus_embeddings.ndim == 2 else None,
            "normalized": True,
            "dtype": "float32",
            "device_used": str(device),
        }
        if device.type == "cuda":
            info["peak_cuda_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            info["peak_cuda_memory_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        write_json(info_path, info)
        print(f"Wrote dense embeddings: {output_dir}")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

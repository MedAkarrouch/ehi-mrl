#!/usr/bin/env python3
"""Benchmark dense bi-encoder batch sizes on H200."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

from data_utils import ensure_dir
from retrieval_utils import load_config, repo_root_from_script, resolve_path
from train_dense_biencoder import PairDataset, make_collate_fn, read_jsonl_texts, read_training_pairs


FIELDNAMES = [
    "batch_size",
    "status",
    "mean_step_time_sec",
    "examples_per_sec",
    "peak_cuda_memory_allocated_gb",
    "peak_cuda_memory_reserved_gb",
    "loss",
    "error",
]


def parse_batch_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got: {value}")


def initialize_csv(path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def cleanup_cuda(torch_module: object) -> None:
    gc.collect()
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
            torch_module.cuda.empty_cache()
    except RuntimeError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--batch-sizes", default="128,256,512,768,1024,1536,2048")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/training_benchmarks/dense_nq320k_h200_batch_sweep.csv"))
    parser.add_argument("--num-workers", type=int, default=2, help="Benchmark DataLoader worker count.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Benchmark DataLoader prefetch factor.")
    parser.add_argument("--persistent-workers", type=parse_bool, default=False, help="Use persistent DataLoader workers.")
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from dense_modeling import in_batch_contrastive_loss, load_tokenizer_and_model, move_batch_to_device

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the H200 batch-size benchmark.")
        repo_root = repo_root_from_script(__file__)
        output_path = resolve_path(repo_root, args.output)
        initialize_csv(output_path)
        config = load_config(resolve_path(repo_root, args.config))
        device = torch.device("cuda")
        if hasattr(torch, "set_float32_matmul_precision") and bool(config.get("tf32", True)):
            torch.set_float32_matmul_precision("high")
        print("torch:", torch.__version__)
        print("gpu:", torch.cuda.get_device_name(0))
        print("bf16 supported:", torch.cuda.is_bf16_supported())
        queries = read_jsonl_texts(resolve_path(repo_root, config["train_queries"]))
        corpus = read_jsonl_texts(resolve_path(repo_root, config["corpus"]), document=True)
        pairs = read_training_pairs(resolve_path(repo_root, config["train_triples"]), queries, corpus)
        tokenizer, model = load_tokenizer_and_model(
            str(config["base_model_name_or_path"]),
            cache_dir=resolve_path(repo_root, config["cache_dir"]),
            normalize_embeddings=bool(config.get("normalize_embeddings", True)),
        )
        model.to(device)
        model.train()
        rows: list[dict[str, object]] = []
        for batch_size in parse_batch_sizes(args.batch_sizes):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            optimizer = None
            dataloader = None
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
            loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": True,
                "num_workers": args.num_workers,
                "pin_memory": bool(config.get("pin_memory", True)),
                "drop_last": True,
                "collate_fn": make_collate_fn(tokenizer, int(config["max_query_length"]), int(config["max_doc_length"])),
            }
            if loader_kwargs["num_workers"] > 0:
                loader_kwargs["persistent_workers"] = args.persistent_workers
                loader_kwargs["prefetch_factor"] = args.prefetch_factor
            dataloader = DataLoader(PairDataset(pairs), **loader_kwargs)
            step_times: list[float] = []
            total_examples = 0
            last_loss = 0.0
            status = "success"
            error = ""
            try:
                for step, batch in enumerate(dataloader, start=1):
                    if step > args.steps:
                        break
                    start = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    query_batch = move_batch_to_device(batch["query"], device)
                    doc_batch = move_batch_to_device(batch["doc"], device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=bool(config.get("bf16", True)) and torch.cuda.is_bf16_supported()):
                        loss = in_batch_contrastive_loss(
                            model(**query_batch),
                            model(**doc_batch),
                            float(config["temperature"]),
                            bidirectional=bool(config.get("bidirectional_loss", False)),
                        )
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    total_examples += int(batch["size"])
                    last_loss = float(loss.detach().item())
                    step_times.append(time.perf_counter() - start)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    status = "cuda_oom"
                    error = str(exc).splitlines()[0]
                else:
                    raise
            mean_time = sum(step_times) / len(step_times) if step_times else 0.0
            rows.append(
                {
                    "batch_size": batch_size,
                    "status": status,
                    "mean_step_time_sec": mean_time,
                    "examples_per_sec": total_examples / sum(step_times) if step_times and sum(step_times) else 0.0,
                    "peak_cuda_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                    "peak_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
                    "loss": last_loss,
                    "error": error,
                }
            )
            print(rows[-1])
            write_rows(output_path, rows)
            del dataloader
            del optimizer
            cleanup_cuda(torch)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a rigid dense dual encoder with in-batch contrastive learning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

from data_utils import ensure_dir, iter_jsonl, safe_text
from retrieval_utils import build_document_text, load_config, repo_root_from_script, resolve_path


def read_jsonl_texts(path: Path, document: bool = False) -> dict[str, str]:
    rows: dict[str, str] = {}
    for row in iter_jsonl(path):
        row_id = safe_text(row.get("_id"))
        rows[row_id] = build_document_text(row) if document else safe_text(row.get("text"))
    return rows


def read_training_pairs(triples_path: Path, queries: dict[str, str], corpus: dict[str, str]) -> list[tuple[str, str, str, str | None]]:
    pairs: list[tuple[str, str, str, str | None]] = []
    with triples_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            query_id = safe_text(row.get("query-id"))
            positive_id = safe_text(row.get("positive-doc-id"))
            negative_id = safe_text(row.get("negative-doc-id"))
            if query_id in queries and positive_id in corpus:
                pairs.append((query_id, queries[query_id], corpus[positive_id], negative_id or None))
    if not pairs:
        raise RuntimeError("No valid NQ320K training pairs were loaded.")
    return pairs


class PairDataset:
    def __init__(self, pairs: list[tuple[str, str, str, str | None]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[str, str, str, str | None]:
        return self.pairs[index]


def make_collate_fn(tokenizer: Any, max_query_length: int, max_doc_length: int) -> Any:
    def collate(rows: list[tuple[str, str, str, str | None]]) -> dict[str, Any]:
        seen_queries: set[str] = set()
        seen_docs: set[str] = set()
        query_texts: list[str] = []
        doc_texts: list[str] = []
        for query_id, query_text, doc_text, _negative_id in rows:
            doc_key = doc_text[:256]
            if query_id in seen_queries or doc_key in seen_docs:
                continue
            seen_queries.add(query_id)
            seen_docs.add(doc_key)
            query_texts.append(query_text)
            doc_texts.append(doc_text)
        if not query_texts:
            query_texts = [rows[0][1]]
            doc_texts = [rows[0][2]]
        return {
            "query": tokenizer(query_texts, truncation=True, padding=True, max_length=max_query_length, return_tensors="pt"),
            "doc": tokenizer(doc_texts, truncation=True, padding=True, max_length=max_doc_length, return_tensors="pt"),
            "size": len(query_texts),
        }

    return collate


def json_dump(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--override-batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
        from dense_modeling import in_batch_contrastive_loss, load_tokenizer_and_model, move_batch_to_device, save_dense_model

        repo_root = repo_root_from_script(__file__)
        config = load_config(resolve_path(repo_root, args.config))
        random.seed(int(config.get("seed", 42)))
        torch.manual_seed(int(config.get("seed", 42)))
        if hasattr(torch, "set_float32_matmul_precision") and bool(config.get("tf32", True)):
            torch.set_float32_matmul_precision("high")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the H200 dense training job.")
        device = torch.device("cuda")
        print("torch:", torch.__version__)
        print("cuda available:", torch.cuda.is_available())
        print("gpu:", torch.cuda.get_device_name(0))
        print("total vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
        print("bf16 supported:", torch.cuda.is_bf16_supported())

        output_dir = resolve_path(repo_root, config["output_dir"])
        ensure_dir(output_dir)
        config_copy_path = output_dir / "training_config.yaml"
        config_copy_path.write_text(resolve_path(repo_root, args.config).read_text(encoding="utf-8"), encoding="utf-8")

        queries = read_jsonl_texts(resolve_path(repo_root, config["train_queries"]))
        corpus = read_jsonl_texts(resolve_path(repo_root, config["corpus"]), document=True)
        pairs = read_training_pairs(resolve_path(repo_root, config["train_triples"]), queries, corpus)
        tokenizer, model = load_tokenizer_and_model(
            str(config["base_model_name_or_path"]),
            cache_dir=resolve_path(repo_root, config["cache_dir"]),
            normalize_embeddings=bool(config.get("normalize_embeddings", True)),
        )
        if args.resume_from_checkpoint:
            tokenizer, model = load_tokenizer_and_model(str(resolve_path(repo_root, args.resume_from_checkpoint)))
        model.to(device)
        model.train()

        batch_size = int(args.override_batch_size or config["per_device_train_batch_size"])
        print("configured batch size:", batch_size)
        print("max query/doc lengths:", config["max_query_length"], config["max_doc_length"])
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": int(config.get("num_workers", 0)),
            "pin_memory": bool(config.get("pin_memory", True)),
            "drop_last": True,
            "collate_fn": make_collate_fn(tokenizer, int(config["max_query_length"]), int(config["max_doc_length"])),
        }
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
            loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 4))
        dataloader = DataLoader(PairDataset(pairs), **loader_kwargs)
        total_steps = args.max_steps or config.get("max_steps")
        if total_steps is None:
            total_steps = math.ceil(len(dataloader) * int(config["num_train_epochs"]) / int(config["gradient_accumulation_steps"]))
        total_steps = int(total_steps)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
        warmup_steps = int(total_steps * float(config.get("warmup_ratio", 0.0)))
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=bool(config.get("fp16", False)))
        use_bf16 = bool(config.get("bf16", True)) and torch.cuda.is_bf16_supported()
        log_path = output_dir / "train_log.jsonl"
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        step = 0
        examples = 0
        final_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        while step < total_steps:
            for batch in dataloader:
                step_start = time.perf_counter()
                query_batch = move_batch_to_device(batch["query"], device)
                doc_batch = move_batch_to_device(batch["doc"], device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                    query_embeddings = model(**query_batch)
                    doc_embeddings = model(**doc_batch)
                    loss = in_batch_contrastive_loss(
                        query_embeddings,
                        doc_embeddings,
                        float(config["temperature"]),
                        bidirectional=bool(config.get("bidirectional_loss", False)),
                    )
                    loss = loss / int(config["gradient_accumulation_steps"])
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % int(config["gradient_accumulation_steps"]) == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                step += 1
                examples += int(batch["size"])
                final_loss = float(loss.detach().item() * int(config["gradient_accumulation_steps"]))
                if step % int(config["log_every_steps"]) == 0 or step == 1:
                    elapsed = time.perf_counter() - start_time
                    record = {
                        "step": step,
                        "loss": final_loss,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "examples_per_second": examples / elapsed if elapsed else 0.0,
                        "step_time_sec": time.perf_counter() - step_start,
                    }
                    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(json.dumps(record) + "\n")
                    print(record)
                if step % int(config["save_every_steps"]) == 0:
                    checkpoint_dir = output_dir / "checkpoints" / f"step_{step}"
                    save_dense_model(model, tokenizer, checkpoint_dir)
                if step >= total_steps:
                    break
        final_dir = output_dir / "final"
        save_dense_model(model, tokenizer, final_dir)
        total_seconds = time.perf_counter() - start_time
        metrics = {
            "total_steps": step,
            "epochs": config["num_train_epochs"],
            "final_loss": final_loss,
            "last_moving_average_loss": final_loss,
            "total_training_seconds": total_seconds,
            "examples_per_second": examples / total_seconds if total_seconds else 0.0,
            "peak_cuda_memory_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "peak_cuda_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
            "config_path": str(args.config),
            "base_model": config["base_model_name_or_path"],
            "method_label": config["method_label"],
        }
        json_dump(output_dir / "training_metrics.json", metrics)
        print(metrics)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Inspect either a Hugging Face dataset sample or normalized processed files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from data_utils import count_jsonl, count_tsv_rows, iter_jsonl, load_yaml, read_tsv, set_hf_cache


def stream_sample(dataset_id: str, cache_dir: Path) -> tuple[list[str], dict[str, Any] | None]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required; install project dependencies first.") from exc

    try:
        dataset = load_dataset(dataset_id, cache_dir=str(cache_dir), streaming=True)
        if hasattr(dataset, "keys"):
            splits = list(dataset.keys())
            if not splits:
                return [], None
            selected_split = "train" if "train" in dataset else splits[0]
            return splits, next(iter(dataset[selected_split].take(1)), None)
        return [], next(iter(dataset.take(1)), None)
    except Exception as exc:
        raise RuntimeError(f"Unable to inspect Hugging Face dataset '{dataset_id}': {exc}") from exc


def first_jsonl_row(path: Path) -> dict[str, Any] | None:
    return next(iter(iter_jsonl(path)), None)


def inspect_processed(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    if not output_dir.is_dir():
        raise RuntimeError(f"Processed data directory does not exist: {output_dir}")

    files = sorted(path for path in output_dir.iterdir() if path.is_file())
    print(f"dataset_name: {config['dataset_name']}")
    print(f"processed directory: {output_dir}")
    print("files present:")
    for path in files:
        if path.suffix == ".jsonl":
            print(f"  {path.name}: {count_jsonl(path)} rows")
        elif path.suffix == ".tsv":
            print(f"  {path.name}: {count_tsv_rows(path)} rows")
        else:
            print(f"  {path.name}")

    corpus_path = output_dir / "corpus.jsonl"
    query_paths = sorted(output_dir.glob("queries_*.jsonl"))
    qrels_paths = sorted(output_dir.glob("qrels_*.tsv"))
    triples_path = output_dir / "triples_train.tsv"
    print(f"example corpus row: {first_jsonl_row(corpus_path) if corpus_path.is_file() else None}")
    print(f"example query row: {first_jsonl_row(query_paths[0]) if query_paths else None}")
    if qrels_paths:
        _header, rows = read_tsv(qrels_paths[0])
        qrels_example: list[str] | None = rows[0] if rows else None
    else:
        qrels_example = None
    print(f"example qrels row: {qrels_example}")
    if triples_path.is_file():
        _header, rows = read_tsv(triples_path)
        triple_example: list[str] | None = rows[0] if rows else None
        print(f"example triple row: {triple_example}")


def inspect_huggingface(config: dict[str, Any]) -> None:
    cache_dir = set_hf_cache(config["cache_dir"])
    splits, sample = stream_sample(config["hf_dataset"], cache_dir)
    print(f"dataset_name: {config['dataset_name']}")
    print(f"source: {config['source']}")
    print(f"task: {config['task']}")
    print(f"Hugging Face dataset id: {config['hf_dataset']}")
    print(f"available splits: {', '.join(splits) if splits else 'not reported'}")
    print(f"example row from main dataset: {sample}")

    qrels_id = config.get("hf_qrels")
    if qrels_id:
        try:
            _qrels_splits, qrels_sample = stream_sample(qrels_id, cache_dir)
            print(f"example row from qrels: {qrels_sample}")
        except RuntimeError as exc:
            print(f"qrels inspection skipped: {exc}", file=sys.stderr)
    else:
        print("example row from qrels: not configured")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a dataset YAML config.")
    parser.add_argument("--processed", action="store_true", help="Inspect normalized files in the configured output directory.")
    args = parser.parse_args()

    try:
        config = load_yaml(args.config)
        required = ("dataset_name", "source", "hf_dataset", "task", "output_dir", "cache_dir")
        missing = [key for key in required if key not in config]
        if missing:
            raise RuntimeError(f"Config '{args.config}' is missing required keys: {', '.join(missing)}")
        if args.processed:
            inspect_processed(config)
        else:
            inspect_huggingface(config)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

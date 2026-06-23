#!/usr/bin/env python3
"""Inspect either a Hugging Face dataset sample or normalized processed files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from data_utils import count_jsonl, count_tsv_rows, iter_jsonl, load_yaml, read_tsv, set_hf_cache


def truncate_display(value: Any, max_chars: int = 500) -> Any:
    """Recursively shorten strings for terminal display without changing source data."""
    if isinstance(value, str):
        if len(value) <= max_chars:
            return value
        omitted = len(value) - max_chars
        return f"{value[:max_chars]}... [truncated {omitted} characters]"
    if isinstance(value, dict):
        return {key: truncate_display(item, max_chars) for key, item in value.items()}
    if isinstance(value, list):
        return [truncate_display(item, max_chars) for item in value]
    if isinstance(value, tuple):
        return tuple(truncate_display(item, max_chars) for item in value)
    return value


def stream_sample(
    dataset_id: str, cache_dir: Path, config_name: str | None = None, split_name: str | None = None
) -> tuple[list[str], dict[str, Any] | None]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required; install project dependencies first.") from exc

    try:
        if config_name is not None:
            dataset = load_dataset(
                dataset_id,
                config_name,
                split=split_name,
                cache_dir=str(cache_dir),
                streaming=True,
            )
            return [split_name or "default"], next(iter(dataset.take(1)), None)

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


def inspect_processed(config: dict[str, Any], max_display_chars: int) -> None:
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
    corpus_example = first_jsonl_row(corpus_path) if corpus_path.is_file() else None
    query_example = first_jsonl_row(query_paths[0]) if query_paths else None
    print(f"example corpus row: {truncate_display(corpus_example, max_display_chars)}")
    print(f"example query row: {truncate_display(query_example, max_display_chars)}")
    if qrels_paths:
        _header, rows = read_tsv(qrels_paths[0])
        qrels_example: list[str] | None = rows[0] if rows else None
    else:
        qrels_example = None
    print(f"example qrels row: {truncate_display(qrels_example, max_display_chars)}")
    if triples_path.is_file():
        _header, rows = read_tsv(triples_path)
        triple_example: list[str] | None = rows[0] if rows else None
        print(f"example triple row: {truncate_display(triple_example, max_display_chars)}")


def inspect_huggingface(config: dict[str, Any], max_display_chars: int) -> None:
    cache_dir = set_hf_cache(config["cache_dir"])
    corpus_config = config.get("hf_corpus_config")
    corpus_split = config.get("hf_corpus_split")
    splits, sample = stream_sample(config["hf_dataset"], cache_dir, corpus_config, corpus_split)
    print(f"dataset_name: {config['dataset_name']}")
    print(f"source: {config['source']}")
    print(f"task: {config['task']}")
    print(f"Hugging Face dataset id: {config['hf_dataset']}")
    print(f"available splits: {', '.join(splits) if splits else 'not reported'}")
    print(f"example row from main dataset: {truncate_display(sample, max_display_chars)}")

    qrels_id = config.get("hf_qrels")
    if qrels_id:
        try:
            _qrels_splits, qrels_sample = stream_sample(qrels_id, cache_dir)
            print(f"example row from qrels: {truncate_display(qrels_sample, max_display_chars)}")
        except RuntimeError as exc:
            print(f"qrels inspection skipped: {exc}", file=sys.stderr)
    else:
        print("example row from qrels: not configured")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a dataset YAML config.")
    parser.add_argument("--processed", action="store_true", help="Inspect normalized files in the configured output directory.")
    parser.add_argument(
        "--max-display-chars",
        type=int,
        default=500,
        help="Maximum number of characters shown for each displayed string value (default: 500).",
    )
    args = parser.parse_args()

    if args.max_display_chars < 1:
        parser.error("--max-display-chars must be a positive integer.")

    try:
        config = load_yaml(args.config)
        required = ("dataset_name", "source", "hf_dataset", "task", "output_dir", "cache_dir")
        missing = [key for key in required if key not in config]
        if missing:
            raise RuntimeError(f"Config '{args.config}' is missing required keys: {', '.join(missing)}")
        if args.processed:
            inspect_processed(config, args.max_display_chars)
        else:
            inspect_huggingface(config, args.max_display_chars)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

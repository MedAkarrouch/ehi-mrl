#!/usr/bin/env python3
"""Print small streamed samples from a configured Hugging Face dataset."""

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def load_config(config_path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install project dependencies first.") from exc

    try:
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
    except OSError as exc:
        raise RuntimeError(f"Could not read config '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in '{config_path}': {exc}") from exc
    if not isinstance(config, dict):
        raise RuntimeError(f"Config '{config_path}' must contain a YAML mapping.")
    return config


def configure_hf_cache(cache_dir: Path) -> Path:
    cache_dir = cache_dir.resolve()
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
    return cache_dir


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a dataset YAML config.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        required = ("dataset_name", "source", "hf_dataset", "task", "cache_dir")
        missing = [key for key in required if key not in config]
        if missing:
            raise RuntimeError(f"Config '{args.config}' is missing required keys: {', '.join(missing)}")

        cache_dir = configure_hf_cache(Path(config["cache_dir"]))
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
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

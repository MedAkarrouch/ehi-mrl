#!/usr/bin/env python3
"""Inspect a configured Hugging Face dataset without materializing it."""

import argparse
import json
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
    datasets_cache = cache_dir / "datasets"
    transformers_cache = cache_dir / "transformers"
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)
    transformers_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    return cache_dir


def stream_dataset_metadata(dataset_id: str, cache_dir: Path) -> tuple[list[str], dict[str, Any] | None]:
    """Read at most one streamed row, avoiding full dataset materialization."""
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
            sample = next(iter(dataset[selected_split].take(1)), None)
            return splits, sample

        sample = next(iter(dataset.take(1)), None)
        return [], sample
    except Exception as exc:
        raise RuntimeError(f"Unable to inspect Hugging Face dataset '{dataset_id}': {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a dataset YAML config.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        required = ("dataset_name", "source", "hf_dataset", "task", "output_dir", "cache_dir")
        missing = [key for key in required if key not in config]
        if missing:
            raise RuntimeError(f"Config '{args.config}' is missing required keys: {', '.join(missing)}")

        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = configure_hf_cache(Path(config["cache_dir"]))

        info: dict[str, Any] = {
            key: config[key]
            for key in ("dataset_name", "source", "hf_dataset", "task", "output_dir", "cache_dir")
        }
        if "hf_qrels" in config:
            info["hf_qrels"] = config["hf_qrels"]

        try:
            splits, _sample = stream_dataset_metadata(config["hf_dataset"], cache_dir)
            info["available_splits"] = splits
            info["status"] = "metadata_loaded"
        except RuntimeError as exc:
            info["available_splits"] = []
            info["status"] = f"error: {exc}"
            print(f"error: {exc}", file=sys.stderr)
            (output_dir / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
            return 1

        info_path = output_dir / "dataset_info.json"
        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote dataset metadata to {info_path}")
        print(f"Detected splits: {', '.join(splits) if splits else 'not reported'}")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Print the configured BEIR Hugging Face calls without downloading data by default."""

from __future__ import annotations

import argparse
from pathlib import Path

from data_utils import load_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a BEIR dataset YAML config.")
    parser.add_argument("--load-sample", action="store_true", help="Load one streamed corpus row after printing the planned calls.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    required = ("hf_dataset", "hf_corpus_config", "hf_corpus_split", "hf_queries_config", "hf_queries_split", "hf_qrels")
    missing = [key for key in required if key not in config]
    if missing:
        raise RuntimeError(f"Config is missing required BEIR keys: {', '.join(missing)}")

    print(f"dataset id: {config['hf_dataset']}")
    print(
        "corpus call: "
        f"load_dataset({config['hf_dataset']!r}, {config['hf_corpus_config']!r}, split={config['hf_corpus_split']!r}, cache_dir=...)"
    )
    print(
        "queries call: "
        f"load_dataset({config['hf_dataset']!r}, {config['hf_queries_config']!r}, split={config['hf_queries_split']!r}, cache_dir=...)"
    )
    print(f"qrels dataset: {config['hf_qrels']}")
    print("qrels split: auto-detect (test > validation > dev > train > first available)")

    if args.load_sample:
        from inspect_dataset import stream_sample
        from data_utils import set_hf_cache

        _splits, sample = stream_sample(
            config["hf_dataset"],
            set_hf_cache(config["cache_dir"]),
            config["hf_corpus_config"],
            config["hf_corpus_split"],
        )
        print(f"sample corpus row: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

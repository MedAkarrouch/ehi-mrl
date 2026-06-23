#!/usr/bin/env python3
"""Offline validation for the dataset configuration files."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "data_beir_scifact.yaml": True,
    "data_beir_fiqa.yaml": True,
    "data_nq320k.yaml": False,
}
REQUIRED_KEYS = ("dataset_name", "source", "hf_dataset", "task", "output_dir", "cache_dir")
NQ320K_KEYS = ("hf_corpus_config", "hf_pairs_config", "hf_corpus_split", "hf_train_split", "hf_dev_split", "random_seed")


def contains_key(text: str, key: str) -> bool:
    return re.search(rf"^{re.escape(key)}:\s*\S+", text, flags=re.MULTILINE) is not None


def main() -> None:
    for filename, requires_qrels in CONFIGS.items():
        path = ROOT / "configs" / filename
        assert path.is_file(), f"Missing configuration file: {path}"
        text = path.read_text(encoding="utf-8")
        for key in REQUIRED_KEYS:
            assert contains_key(text, key), f"{filename} is missing required key: {key}"
        if requires_qrels:
            assert contains_key(text, "hf_qrels"), f"{filename} is missing required key: hf_qrels"
            assert contains_key(text, "random_seed"), f"{filename} is missing required key: random_seed"
        else:
            for key in NQ320K_KEYS:
                assert contains_key(text, key), f"{filename} is missing required key: {key}"
    print("Dataset configuration checks passed.")


if __name__ == "__main__":
    main()

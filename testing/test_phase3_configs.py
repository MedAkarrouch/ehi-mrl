#!/usr/bin/env python3
"""Offline checks for Phase 3 FAISS IVF configs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_utils import load_config  # noqa: E402


EXPECTED_PRIMARY_METRICS = {
    "configs/faiss_ivf_nq320k.yaml": ["Hit@1", "MRR@10", "Recall@10", "Recall@100"],
    "configs/faiss_ivf_scifact.yaml": ["nDCG@10", "Recall@100", "MRR@10", "Hit@1"],
    "configs/faiss_ivf_fiqa.yaml": ["nDCG@10", "Recall@100", "MRR@10", "Hit@1"],
}


REQUIRED_KEYS = {
    "dataset_name",
    "split",
    "embedding_dir",
    "index_dir",
    "results_dir",
    "metric",
    "top_k",
    "nlist_values",
    "nprobe_values",
    "primary_metrics",
}


def main() -> None:
    for relative_path, expected_metrics in EXPECTED_PRIMARY_METRICS.items():
        path = ROOT / relative_path
        assert path.is_file(), f"Missing FAISS IVF config: {path}"
        config = load_config(path)
        assert REQUIRED_KEYS.issubset(config), f"{path} missing required keys"
        assert (ROOT / str(config["exact_baseline_config"])).is_file(), config["exact_baseline_config"]
        assert config["metric"] == "inner_product"
        assert config["primary_metrics"] == expected_metrics
        assert all(isinstance(value, int) and value > 0 for value in config["nlist_values"]), config["nlist_values"]
        assert all(isinstance(value, int) and value > 0 for value in config["nprobe_values"]), config["nprobe_values"]
    print("Phase 3 FAISS config checks passed.")


if __name__ == "__main__":
    main()

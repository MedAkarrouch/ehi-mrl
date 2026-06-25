#!/usr/bin/env python3
"""Offline checks for Phase 2 exact baseline configs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_utils import load_config  # noqa: E402


EXPECTED_PRIMARY_METRICS = {
    "configs/exact_baseline_nq320k.yaml": ["Hit@1", "MRR@10", "Recall@10", "Recall@100"],
    "configs/exact_baseline_scifact.yaml": ["nDCG@10", "Recall@100", "MRR@10", "Hit@1"],
    "configs/exact_baseline_fiqa.yaml": ["nDCG@10", "Recall@100", "MRR@10", "Hit@1"],
}


def main() -> None:
    encoder_path = ROOT / "configs" / "encoder_sbert_distilbert_nli_stsb.yaml"
    assert encoder_path.is_file(), f"Missing encoder config: {encoder_path}"
    encoder_config = load_config(encoder_path)
    assert encoder_config["hf_model_name"] == "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
    assert encoder_config["encoder_name"] == "sbert_distilbert_nli_stsb"

    for relative_path, expected_metrics in EXPECTED_PRIMARY_METRICS.items():
        path = ROOT / relative_path
        assert path.is_file(), f"Missing exact baseline config: {path}"
        config = load_config(path)
        assert (ROOT / str(config["dataset_config"])).is_file(), config["dataset_config"]
        assert (ROOT / str(config["encoder_config"])).is_file(), config["encoder_config"]
        assert config["encoder_config"] == "configs/encoder_sbert_distilbert_nli_stsb.yaml"
        assert config["primary_metrics"] == expected_metrics
        assert "query_batch_size" in config
        assert "corpus_chunk_size" in config
        assert "device" in config
        assert int(config["query_batch_size"]) > 0
        assert int(config["corpus_chunk_size"]) > 0
    print("Phase 2 config checks passed.")


if __name__ == "__main__":
    main()

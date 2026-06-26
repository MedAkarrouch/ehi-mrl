#!/usr/bin/env python3
"""Offline checks for Phase 4 fine-tuned dense configs and jobs."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from retrieval_utils import load_config  # noqa: E402


PHASE4_CONFIGS = [
    "configs/train_dense_nq320k_distilbert.yaml",
    "configs/exact_finetuned_dense_nq320k.yaml",
    "configs/exact_finetuned_dense_scifact.yaml",
    "configs/exact_finetuned_dense_fiqa.yaml",
    "configs/faiss_ivf_finetuned_dense_nq320k.yaml",
    "configs/faiss_ivf_finetuned_dense_scifact.yaml",
    "configs/faiss_ivf_finetuned_dense_fiqa.yaml",
]


def main() -> None:
    for relative_path in PHASE4_CONFIGS:
        path = ROOT / relative_path
        assert path.is_file(), f"Missing Phase 4 config: {path}"
        text = path.read_text(encoding="utf-8")
        assert "Hit@1" not in text, f"Phase 4 config must not include Hit@1: {path}"
        assert "Phase 4 exact" not in text
        assert "Phase 4 dense" not in text
        assert "Phase 4 FAISS" not in text
        config = load_config(path)
        if "method_label" in config:
            assert config["method_label"] in {
                "Fine-tuned Dense",
                "Fine-tuned Dense + Exact Search",
                "Fine-tuned Dense + FAISS-IVF",
            }
        if "embedding_dir" in config:
            assert "fine_tuned_dense_nq320k_distilbert" in str(config["embedding_dir"])
        if "results_dir" in config:
            assert "fine_tuned_dense_nq320k_distilbert" in str(config["results_dir"])
    train_job = (ROOT / "jobs" / "train_dense_nq320k_h200.sbatch").read_text(encoding="utf-8")
    bench_job = (ROOT / "jobs" / "benchmark_dense_batch_h200.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:h200:1" in train_job
    assert "#SBATCH --gres=gpu:h200:1" in bench_job
    assert "torch.amp.autocast" in (ROOT / "scripts" / "train_dense_biencoder.py").read_text(encoding="utf-8")
    assert "max_memory_allocated" in (ROOT / "scripts" / "train_dense_biencoder.py").read_text(encoding="utf-8")
    print("Phase 4 config and H200 job checks passed.")


if __name__ == "__main__":
    main()

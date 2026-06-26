#!/usr/bin/env python3
"""Syntax checks for Phase 4 dense training scripts."""

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/dense_modeling.py",
    "scripts/train_dense_biencoder.py",
    "scripts/benchmark_dense_batch_size.py",
    "scripts/embed_dense_model.py",
    "scripts/run_finetuned_dense_exact.py",
    "scripts/run_finetuned_dense_faiss.py",
)


def main() -> None:
    for relative_path in FILES:
        path = ROOT / relative_path
        assert path.is_file(), f"Missing Phase 4 script: {path}"
        py_compile.compile(str(path), doraise=True)
    print("Phase 4 training script compile checks passed.")


if __name__ == "__main__":
    main()

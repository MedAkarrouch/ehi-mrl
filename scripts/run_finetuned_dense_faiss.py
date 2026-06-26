#!/usr/bin/env python3
"""Run FAISS-IVF sweeps for fine-tuned dense embeddings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from retrieval_utils import repo_root_from_script, resolve_path


CONFIGS = [
    "configs/faiss_ivf_finetuned_dense_nq320k.yaml",
    "configs/faiss_ivf_finetuned_dense_scifact.yaml",
    "configs/faiss_ivf_finetuned_dense_fiqa.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    repo_root = repo_root_from_script(__file__)
    try:
        for config in CONFIGS:
            command = [
                sys.executable,
                str(repo_root / "scripts" / "run_faiss_ivf_sweep.py"),
                "--config",
                str(resolve_path(repo_root, config)),
            ]
            if args.overwrite:
                command.append("--overwrite")
            print(" ".join(command))
            completed = subprocess.run(command, cwd=repo_root, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"FAISS sweep failed for {config} with exit code {completed.returncode}.")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

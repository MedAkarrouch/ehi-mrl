#!/usr/bin/env python3
"""Offline content checks for the source-able HPC environment file."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SNIPPETS = (
    "module purge",
    "module load python/3.11.2",
    "gpt2_cuda_env",
    "PROJECT_ROOT",
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "TRANSFORMERS_CACHE",
)


def main() -> None:
    path = ROOT / "configs" / "hpc_env.sh"
    assert path.is_file(), f"Missing HPC environment file: {path}"
    text = path.read_text(encoding="utf-8")
    for snippet in REQUIRED_SNIPPETS:
        assert snippet in text, f"hpc_env.sh is missing: {snippet}"
    print("HPC environment file checks passed.")


if __name__ == "__main__":
    main()

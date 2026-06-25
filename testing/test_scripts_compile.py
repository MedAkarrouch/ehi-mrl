#!/usr/bin/env python3
"""Offline syntax validation for Phase 0 and Phase 1 scripts."""

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/data_utils.py",
    "scripts/debug_beir_loading.py",
    "scripts/prepare_data.py",
    "scripts/inspect_dataset.py",
    "scripts/validate_processed_data.py",
    "scripts/analyze_qrels.py",
    "scripts/retrieval_utils.py",
    "scripts/embed_dataset.py",
    "scripts/exact_search.py",
    "scripts/evaluate_run.py",
    "scripts/run_exact_baseline.py",
    "scripts/install_project_deps.sh",
)


def main() -> None:
    for relative_path in FILES:
        path = ROOT / relative_path
        assert path.is_file(), f"Missing required script: {path}"

    for relative_path in (path for path in FILES if path.endswith(".py")):
        py_compile.compile(str(ROOT / relative_path), doraise=True)
    print("Script existence and Python compilation checks passed.")


if __name__ == "__main__":
    main()

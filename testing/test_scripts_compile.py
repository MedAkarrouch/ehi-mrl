#!/usr/bin/env python3
"""Offline syntax validation for Phase 0 scripts."""

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/prepare_data.py",
    "scripts/inspect_dataset.py",
    "scripts/install_project_deps.sh",
)


def main() -> None:
    for relative_path in FILES:
        path = ROOT / relative_path
        assert path.is_file(), f"Missing required script: {path}"

    for relative_path in FILES[:2]:
        py_compile.compile(str(ROOT / relative_path), doraise=True)
    print("Script existence and Python compilation checks passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Check plotting accepts snake_case and display-style efficiency columns."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "nq": ("nq320k", "dev"),
    "scifact": ("beir_scifact", "test"),
    "fiqa": ("beir_fiqa", "test"),
}
METRICS = {"Hit@1": 0.7, "MRR@10": 0.8, "Recall@10": 0.82, "Recall@100": 0.9, "nDCG@10": 0.75}


def plotting_deps_available() -> bool:
    missing = [
        name
        for name in ("matplotlib", "pandas", "numpy")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        print(f"FAISS IVF column-normalization test skipped: missing {', '.join(missing)}.")
        return False
    return True


def write_sweep(path: Path, dataset_name: str, split: str, display_columns: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    docs_column = "%DocsVisited" if display_columns else "percent_docs_visited"
    latency_column = "LatencyMsPerQuery" if display_columns else "latency_ms_per_query"
    avg_column = "AvgDocsVisited" if display_columns else "avg_docs_visited"
    fieldnames = [
        "dataset_name",
        "split",
        "nlist",
        "nprobe",
        avg_column,
        docs_column,
        latency_column,
        "Hit@1",
        "MRR@10",
        "Recall@10",
        "Recall@100",
        "nDCG@10",
    ]
    rows = [
        [dataset_name, split, 16, 1, 10, 2.0, 0.5, 0.5, 0.6, 0.7, 0.8, 0.55],
        [dataset_name, split, 16, 4, 40, 8.0, 0.8, 0.6, 0.7, 0.8, 0.88, 0.68],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def write_exact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": METRICS}), encoding="utf-8")


def run_case(root: Path, display_columns: bool) -> None:
    output_dir = root / ("display" if display_columns else "snake") / "plots"
    paths = {}
    for key, (dataset_name, split) in DATASETS.items():
        sweep = root / ("display" if display_columns else "snake") / key / "sweep_summary.csv"
        exact = root / ("display" if display_columns else "snake") / key / f"metrics_{split}.json"
        write_sweep(sweep, dataset_name, split, display_columns)
        write_exact(exact)
        paths[f"{key}_sweep"] = sweep
        paths[f"{key}_exact"] = exact
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "plot_faiss_ivf_sweeps.py"),
            "--nq-sweep",
            str(paths["nq_sweep"]),
            "--scifact-sweep",
            str(paths["scifact_sweep"]),
            "--fiqa-sweep",
            str(paths["fiqa_sweep"]),
            "--nq-exact",
            str(paths["nq_exact"]),
            "--scifact-exact",
            str(paths["scifact_exact"]),
            "--fiqa-exact",
            str(paths["fiqa_exact"]),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "plot_manifest.json").is_file()


def main() -> None:
    if not plotting_deps_available():
        return
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        run_case(root, display_columns=False)
        run_case(root, display_columns=True)
    print("FAISS IVF plotting column-normalization checks passed.")


if __name__ == "__main__":
    main()

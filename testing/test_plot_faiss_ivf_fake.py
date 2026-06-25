#!/usr/bin/env python3
"""Fake-data subprocess test for FAISS IVF plotting."""

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
METRICS = {
    "Hit@1": 0.70,
    "MRR@10": 0.76,
    "Recall@10": 0.82,
    "Recall@100": 0.90,
    "nDCG@10": 0.74,
}
EXPECTED_PLOTS = [
    "nq320k_recall100_vs_docsvisited",
    "nq320k_recall10_vs_docsvisited",
    "nq320k_mrr10_vs_docsvisited",
    "scifact_ndcg10_vs_docsvisited",
    "scifact_recall100_vs_docsvisited",
    "scifact_mrr10_vs_docsvisited",
    "fiqa_ndcg10_vs_docsvisited",
    "fiqa_recall100_vs_docsvisited",
    "fiqa_mrr10_vs_docsvisited",
]


def plotting_deps_available() -> bool:
    missing = [
        name
        for name in ("matplotlib", "pandas", "numpy")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        print(f"FAISS IVF plotting fake-data test skipped: missing {', '.join(missing)}.")
        return False
    return True


def write_sweep(path: Path, dataset_name: str, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "split",
        "nlist",
        "nprobe",
        "top_k",
        "query_count",
        "corpus_count",
        "avg_docs_visited",
        "percent_docs_visited",
        "search_seconds",
        "latency_ms_per_query",
        "Hit@1",
        "MRR@10",
        "Recall@1",
        "Recall@10",
        "Recall@100",
        "nDCG@10",
    ]
    rows = [
        (dataset_name, split, 32, 1, 100, 10, 1000, 20, 2.0, 0.10, 1.0, 0.50, 0.55, 0.40, 0.60, 0.70, 0.52),
        (dataset_name, split, 32, 4, 100, 10, 1000, 80, 8.0, 0.18, 1.8, 0.62, 0.68, 0.52, 0.74, 0.83, 0.64),
        (dataset_name, split, 64, 8, 100, 10, 1000, 180, 18.0, 0.30, 3.0, 0.67, 0.72, 0.58, 0.79, 0.88, 0.69),
        (dataset_name, split, 64, 16, 100, 10, 1000, 320, 32.0, 0.45, 4.5, 0.69, 0.75, 0.60, 0.81, 0.89, 0.72),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def write_exact(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": METRICS}), encoding="utf-8")


def main() -> None:
    if not plotting_deps_available():
        return
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        output_dir = root / "plots"
        paths = {}
        for key, (dataset_name, split) in DATASETS.items():
            sweep = root / key / "sweep_summary.csv"
            exact = root / key / f"metrics_{split}.json"
            write_sweep(sweep, dataset_name, split)
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
        for stem in EXPECTED_PLOTS:
            for suffix in (".svg", ".png"):
                path = output_dir / f"{stem}{suffix}"
                assert path.is_file(), path
                assert path.stat().st_size > 0, path
        assert (output_dir / "best_operating_points.csv").is_file()
        assert (output_dir / "plot_manifest.json").is_file()
    print("FAISS IVF plotting fake-data checks passed.")


if __name__ == "__main__":
    main()

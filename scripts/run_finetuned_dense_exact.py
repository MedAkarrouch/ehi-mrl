#!/usr/bin/env python3
"""Encode and evaluate Fine-tuned Dense + Exact Search on configured datasets."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from data_utils import ensure_dir
from retrieval_utils import load_config, repo_root_from_script, resolve_path


DATASETS = [
    ("nq320k", "configs/data_nq320k.yaml", "configs/exact_finetuned_dense_nq320k.yaml", "dev"),
    ("beir_scifact", "configs/data_beir_scifact.yaml", "configs/exact_finetuned_dense_scifact.yaml", "test"),
    ("beir_fiqa", "configs/data_beir_fiqa.yaml", "configs/exact_finetuned_dense_fiqa.yaml", "test"),
]


def run_step(name: str, command: list[str], cwd: Path) -> None:
    print("")
    print(f"=== {name} ===")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}.")


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    metric_fields = ["MRR@10", "Recall@10", "Recall@100", "nDCG@10"]
    fieldnames = ["dataset_name", "split", "method_label", *metric_fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    try:
        repo_root = repo_root_from_script(__file__)
        rows: list[dict[str, object]] = []
        for dataset_name, dataset_config, exact_config, split in DATASETS:
            exact_path = resolve_path(repo_root, exact_config)
            exact = load_config(exact_path)
            output_dir = resolve_path(repo_root, exact["embedding_dir"])
            embed_command = [
                sys.executable,
                str(repo_root / "scripts" / "embed_dense_model.py"),
                "--model-dir",
                str(resolve_path(repo_root, args.model_dir)),
                "--dataset-config",
                str(resolve_path(repo_root, dataset_config)),
                "--output-dir",
                str(output_dir),
                "--batch-size",
                str(args.batch_size),
                "--max-length",
                "192",
                "--split",
                split,
            ]
            if args.overwrite:
                embed_command.append("--overwrite")
            run_step(f"Encode {dataset_name}", embed_command, repo_root)

            search_command = [sys.executable, str(repo_root / "scripts" / "exact_search.py"), "--config", str(exact_path)]
            if args.overwrite:
                search_command.append("--overwrite")
            run_step(f"Exact search {dataset_name}", search_command, repo_root)

            eval_command = [sys.executable, str(repo_root / "scripts" / "evaluate_run.py"), "--config", str(exact_path)]
            run_step(f"Evaluate {dataset_name}", eval_command, repo_root)
            metrics_path = resolve_path(repo_root, exact["results_dir"]) / f"metrics_{split}.json"
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)["metrics"]
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "split": split,
                    "method_label": exact["method_label"],
                    **{metric: metrics.get(metric, "") for metric in ["MRR@10", "Recall@10", "Recall@100", "nDCG@10"]},
                }
            )
        write_summary(repo_root / "results" / "exact" / "fine_tuned_dense_nq320k_distilbert_summary.csv", rows)
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

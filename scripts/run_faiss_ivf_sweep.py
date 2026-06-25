#!/usr/bin/env python3
"""Run a FAISS IVF nlist/nprobe sweep over existing Phase 2 embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from data_utils import ensure_dir
from retrieval_utils import load_config, repo_root_from_script, resolve_path


SUMMARY_FIELDS = [
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


def parse_int_values(values: list[str] | None, default_values: list[Any]) -> list[int]:
    if not values:
        return [int(value) for value in default_values]
    parsed: list[int] = []
    for value in values:
        for part in value.split(","):
            stripped = part.strip()
            if stripped:
                parsed.append(int(stripped))
    return parsed


def run_step(name: str, command: list[str], cwd: Path) -> None:
    print("")
    print(f"=== {name} ===")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}.")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path.name} must contain a JSON object.")
    return data


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def print_summary(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No FAISS IVF sweep rows were produced.")
        return
    print("")
    print("FAISS IVF sweep summary:")
    print("nlist\tnprobe\t%DocsVisited\tLatencyMsPerQuery\tHit@1\tMRR@10\tRecall@100\tnDCG@10")
    for row in rows:
        print(
            f"{row['nlist']}\t{row['nprobe']}\t"
            f"{float(row.get('percent_docs_visited', 0.0)):.4f}\t"
            f"{float(row.get('latency_ms_per_query', 0.0)):.4f}\t"
            f"{float(row.get('Hit@1', 0.0)):.6f}\t"
            f"{float(row.get('MRR@10', 0.0)):.6f}\t"
            f"{float(row.get('Recall@100', 0.0)):.6f}\t"
            f"{float(row.get('nDCG@10', 0.0)):.6f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a FAISS IVF YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Phase 3 outputs.")
    parser.add_argument("--skip-build", action="store_true", help="Skip building indexes.")
    parser.add_argument("--skip-search", action="store_true", help="Skip FAISS search.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip relevance evaluation.")
    parser.add_argument("--nlist", action="append", help="Override nlist values; can be repeated or comma-separated.")
    parser.add_argument("--nprobe", action="append", help="Override nprobe values; can be repeated or comma-separated.")
    args = parser.parse_args()

    try:
        repo_root = repo_root_from_script(__file__)
        config_path = resolve_path(repo_root, args.config)
        config = load_config(config_path)
        split = str(config["split"])
        results_dir = resolve_path(repo_root, config["results_dir"])
        exact_config = resolve_path(repo_root, config["exact_baseline_config"])
        nlist_values = parse_int_values(args.nlist, list(config["nlist_values"]))
        nprobe_values = parse_int_values(args.nprobe, list(config["nprobe_values"]))

        rows: list[dict[str, Any]] = []
        for nlist in nlist_values:
            if not args.skip_build:
                build_command = [
                    sys.executable,
                    str(repo_root / "scripts" / "build_faiss_ivf.py"),
                    "--config",
                    str(config_path),
                    "--nlist",
                    str(nlist),
                ]
                if args.overwrite:
                    build_command.append("--overwrite")
                run_step(f"Build IVF nlist={nlist}", build_command, repo_root)

            for nprobe in nprobe_values:
                if nprobe > nlist:
                    print(f"Skipping nlist={nlist} nprobe={nprobe}: nprobe is larger than nlist.")
                    continue
                run_path = results_dir / f"run_{split}_nlist{nlist}_nprobe{nprobe}.tsv"
                search_info_path = results_dir / f"search_info_{split}_nlist{nlist}_nprobe{nprobe}.json"
                metrics_json_path = results_dir / f"metrics_{split}_nlist{nlist}_nprobe{nprobe}.json"
                metrics_csv_path = results_dir / f"metrics_{split}_nlist{nlist}_nprobe{nprobe}.csv"

                if not args.skip_search:
                    search_command = [
                        sys.executable,
                        str(repo_root / "scripts" / "search_faiss_ivf.py"),
                        "--config",
                        str(config_path),
                        "--nlist",
                        str(nlist),
                        "--nprobe",
                        str(nprobe),
                    ]
                    if args.overwrite:
                        search_command.append("--overwrite")
                    run_step(f"Search IVF nlist={nlist} nprobe={nprobe}", search_command, repo_root)

                if not args.skip_eval:
                    eval_command = [
                        sys.executable,
                        str(repo_root / "scripts" / "evaluate_run.py"),
                        "--config",
                        str(exact_config),
                        "--run-file",
                        str(run_path),
                        "--output-json",
                        str(metrics_json_path),
                        "--output-csv",
                        str(metrics_csv_path),
                    ]
                    run_step(f"Evaluate IVF nlist={nlist} nprobe={nprobe}", eval_command, repo_root)

                if search_info_path.is_file() and metrics_json_path.is_file():
                    search_info = read_json(search_info_path)
                    metrics_result = read_json(metrics_json_path)
                    row = {
                        "dataset_name": config["dataset_name"],
                        "split": split,
                        "nlist": nlist,
                        "nprobe": nprobe,
                        "top_k": search_info.get("top_k", config.get("top_k")),
                        "query_count": search_info.get("query_count"),
                        "corpus_count": search_info.get("corpus_count"),
                        "avg_docs_visited": search_info.get("avg_docs_visited"),
                        "percent_docs_visited": search_info.get("percent_docs_visited"),
                        "search_seconds": search_info.get("search_seconds"),
                        "latency_ms_per_query": search_info.get("latency_ms_per_query"),
                    }
                    row.update(metrics_result.get("metrics", {}))
                    rows.append(row)

        summary_path = results_dir / "sweep_summary.csv"
        write_summary_csv(summary_path, rows)
        print_summary(rows)
        print(f"Wrote FAISS IVF sweep summary: {summary_path}")
        return 0
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

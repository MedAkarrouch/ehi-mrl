#!/usr/bin/env python3
"""Evaluate an exact retrieval run against normalized qrels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from data_utils import ensure_dir
from retrieval_utils import (
    compute_retrieval_metrics,
    DEFAULT_RETRIEVAL_METRICS,
    filter_queries_to_qrels_covered,
    load_config,
    load_qrels_tsv,
    load_queries_jsonl,
    load_run_tsv,
    qrels_query_ids_missing_from_queries,
    repo_root_from_script,
    resolve_path,
)


ALL_METRICS = DEFAULT_RETRIEVAL_METRICS


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_summary_csv(path: Path, result: dict[str, Any], metric_names: list[str]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "dataset_name",
        "split",
        "qrels_rows",
        "qrels_covered_queries",
        "query_rows",
        "query_rows_with_no_qrels",
        "evaluated_queries",
        "missing_run_queries",
        *metric_names,
    ]
    row = {
        "dataset_name": result["dataset_name"],
        "split": result["split"],
        **result["diagnostics"],
        **result["metrics"],
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def print_summary(result: dict[str, Any]) -> None:
    print(f"Evaluation for {result['dataset_name']} ({result['split']})")
    print(f"run file: {result['run_file']}")
    diagnostics = result["diagnostics"]
    print(f"qrels rows: {diagnostics['qrels_rows']}")
    print(f"qrels-covered queries: {diagnostics['qrels_covered_queries']}")
    print(f"query rows: {diagnostics['query_rows']}")
    print(f"query rows with no qrels: {diagnostics['query_rows_with_no_qrels']}")
    print(f"evaluated queries: {diagnostics['evaluated_queries']}")
    print(f"missing run queries: {diagnostics['missing_run_queries']}")
    print("")
    print("Primary metrics:")
    for metric_name in result["primary_metrics"]:
        print(f"  {metric_name}: {result['metrics'][metric_name]:.6f}")
    print("")
    print("All computed metrics:")
    for metric_name in result["metric_names"]:
        print(f"  {metric_name}: {result['metrics'][metric_name]:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to an exact baseline YAML config.")
    parser.add_argument("--run-file", type=Path, help="Optional run file path. Defaults to results_dir/run_{split}.tsv.")
    parser.add_argument("--output-json", type=Path, help="Optional metrics JSON path.")
    parser.add_argument("--output-csv", type=Path, help="Optional metrics summary CSV path.")
    args = parser.parse_args()

    try:
        repo_root = repo_root_from_script(__file__)
        config = load_config(resolve_path(repo_root, args.config))
        dataset_config = load_config(resolve_path(repo_root, config["dataset_config"]))

        split = str(config["split"])
        processed_dir = resolve_path(repo_root, dataset_config["output_dir"])
        results_dir = resolve_path(repo_root, config["results_dir"])
        query_path = processed_dir / str(config["query_file"])
        qrels_path = processed_dir / str(config["qrels_file"])
        run_path = resolve_path(repo_root, args.run_file) if args.run_file else results_dir / f"run_{split}.tsv"
        output_json = resolve_path(repo_root, args.output_json) if args.output_json else results_dir / f"metrics_{split}.json"
        output_csv = resolve_path(repo_root, args.output_csv) if args.output_csv else results_dir / "metrics_summary.csv"

        queries = load_queries_jsonl(query_path)
        qrels = load_qrels_tsv(qrels_path)
        run = load_run_tsv(run_path)

        missing_query_ids = qrels_query_ids_missing_from_queries(queries, qrels)
        if missing_query_ids:
            preview = ", ".join(missing_query_ids[:10])
            raise RuntimeError(
                f"{qrels_path.name} references {len(missing_query_ids)} query id(s) missing from {query_path.name}: {preview}"
            )

        qrels_covered_queries = filter_queries_to_qrels_covered(queries, qrels)
        evaluated_query_ids = [query_id for query_id, _text in qrels_covered_queries]
        if not evaluated_query_ids:
            raise RuntimeError("No qrels-covered queries are available for evaluation.")

        missing_run_queries = [query_id for query_id in evaluated_query_ids if query_id not in run]
        if missing_run_queries:
            preview = ", ".join(missing_run_queries[:10])
            raise RuntimeError(f"Run file is missing predictions for {len(missing_run_queries)} qrels-covered query id(s): {preview}")

        metric_names = list(config.get("metrics") or ALL_METRICS)
        unsupported_metrics = [metric_name for metric_name in metric_names if metric_name not in ALL_METRICS]
        if unsupported_metrics:
            raise RuntimeError(f"Unsupported metric requested in config: {', '.join(unsupported_metrics)}")
        metrics = compute_retrieval_metrics(qrels, run, evaluated_query_ids, metric_names=metric_names)
        diagnostics = {
            "qrels_rows": sum(len(doc_scores) for doc_scores in qrels.values()),
            "qrels_covered_queries": len(qrels),
            "query_rows": len(queries),
            "query_rows_with_no_qrels": len(queries) - len(qrels_covered_queries),
            "evaluated_queries": len(evaluated_query_ids),
            "missing_run_queries": len(missing_run_queries),
        }
        primary_metrics = list(config.get("primary_metrics", []))
        for metric_name in primary_metrics:
            if metric_name not in metric_names:
                raise RuntimeError(f"Primary metric '{metric_name}' is not included in requested metrics.")

        result: dict[str, Any] = {
            "dataset_name": config["dataset_name"],
            "split": split,
            "run_file": str(run_path),
            "qrels_file": str(qrels_path),
            "query_file": str(query_path),
            "metric_names": metric_names,
            "primary_metrics": primary_metrics,
            "metrics": metrics,
            "diagnostics": diagnostics,
        }
        print_summary(result)
        write_json(output_json, result)
        write_summary_csv(output_csv, result, metric_names)
        print(f"Wrote metrics JSON: {output_json}")
        print(f"Wrote metrics summary CSV: {output_csv}")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

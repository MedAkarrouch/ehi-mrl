#!/usr/bin/env python3
"""Analyze normalized qrels files without downloading datasets."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from data_utils import ensure_dir, iter_jsonl, load_yaml, safe_text


QRELS_HEADER = ["query-id", "corpus-id", "score"]
SPLITS = ("train", "dev", "test")


def parse_simple_yaml_scalar(value: str) -> Any:
    """Parse the simple scalar values used by project config files.

    This fallback keeps the diagnostic runnable in minimal Codex environments
    where PyYAML is not installed. Full YAML parsing still uses PyYAML when it
    is available through ``data_utils.load_yaml``.
    """
    cleaned = value.strip()
    if not cleaned:
        return ""
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        return cleaned[1:-1]
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(cleaned)
    except ValueError:
        return cleaned


def load_simple_yaml_mapping(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if line[:1].isspace():
                    continue
                if ":" not in stripped:
                    raise RuntimeError(f"Invalid simple YAML in '{path}' at line {line_number}: expected key: value.")
                key, value = stripped.split(":", 1)
                key = key.strip()
                if not key:
                    raise RuntimeError(f"Invalid simple YAML in '{path}' at line {line_number}: empty key.")
                data[key] = parse_simple_yaml_scalar(value)
    except OSError as exc:
        raise RuntimeError(f"Could not read YAML file '{path}': {exc}") from exc
    if not data:
        raise RuntimeError(f"YAML file '{path}' must contain a mapping.")
    return data


def load_dataset_config(path: Path) -> dict[str, Any]:
    try:
        return load_yaml(path)
    except RuntimeError as exc:
        if "PyYAML is required" not in str(exc):
            raise
        return load_simple_yaml_mapping(path)


def read_jsonl_ids(path: Path, kind: str) -> tuple[int, set[str], list[str]]:
    errors: list[str] = []
    ids: set[str] = set()
    count = 0
    if not path.is_file():
        return 0, ids, [f"Missing {kind} file: {path.name}"]
    try:
        for line_number, row in enumerate(iter_jsonl(path), start=1):
            count += 1
            identifier = safe_text(row.get("_id"))
            if not identifier:
                errors.append(f"{path.name}:{line_number} has an empty _id.")
                continue
            if identifier in ids:
                errors.append(f"{path.name}:{line_number} duplicates _id '{identifier}'.")
            ids.add(identifier)
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
    return count, ids, errors


def read_qrels(path: Path) -> tuple[int, dict[str, set[str]], set[str], list[str]]:
    errors: list[str] = []
    query_to_doc_ids: dict[str, set[str]] = defaultdict(set)
    doc_ids: set[str] = set()
    row_count = 0
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            try:
                header = next(reader)
            except StopIteration:
                return row_count, query_to_doc_ids, doc_ids, [f"{path.name} is empty; expected header {QRELS_HEADER}."]
            if header != QRELS_HEADER:
                return row_count, query_to_doc_ids, doc_ids, [
                    f"{path.name} header must be {QRELS_HEADER}; found {header}."
                ]
            for line_number, row in enumerate(reader, start=2):
                if not row:
                    continue
                row_count += 1
                if len(row) != 3:
                    errors.append(f"{path.name}:{line_number} must contain exactly 3 columns.")
                    continue
                query_id, corpus_id, _score = (safe_text(value) for value in row)
                if not query_id:
                    errors.append(f"{path.name}:{line_number} has an empty query id.")
                    continue
                if not corpus_id:
                    errors.append(f"{path.name}:{line_number} has an empty corpus id.")
                    continue
                query_to_doc_ids[query_id].add(corpus_id)
                doc_ids.add(corpus_id)
    except OSError as exc:
        errors.append(f"Could not open {path.name}: {exc}")
    return row_count, query_to_doc_ids, doc_ids, errors


def distribution_summary(query_to_doc_ids: dict[str, set[str]], show_top_queries: int) -> dict[str, Any]:
    counts = {query_id: len(doc_ids) for query_id, doc_ids in query_to_doc_ids.items()}
    values = list(counts.values())
    if values:
        minimum = min(values)
        maximum = max(values)
        mean = float(statistics.mean(values))
        median = float(statistics.median(values))
    else:
        minimum = maximum = 0
        mean = median = 0.0

    sorted_queries = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:show_top_queries]
    return {
        "min_relevant_docs_per_qrels_query": minimum,
        "max_relevant_docs_per_qrels_query": maximum,
        "mean_relevant_docs_per_qrels_query": mean,
        "median_relevant_docs_per_qrels_query": median,
        "queries_with_exactly_1_relevant_doc": sum(1 for value in values if value == 1),
        "queries_with_exactly_2_relevant_docs": sum(1 for value in values if value == 2),
        "queries_with_exactly_3_relevant_docs": sum(1 for value in values if value == 3),
        "queries_with_4_or_more_relevant_docs": sum(1 for value in values if value >= 4),
        "top_queries": [
            {
                "query_id": query_id,
                "relevant_docs": count,
                "doc_ids": sorted(query_to_doc_ids[query_id]),
            }
            for query_id, count in sorted_queries
        ],
    }


def analyze_split(
    output_dir: Path,
    split: str,
    corpus_row_count: int,
    corpus_ids: set[str],
    show_top_queries: int,
) -> tuple[dict[str, Any], list[str]]:
    qrels_path = output_dir / f"qrels_{split}.tsv"
    query_path = output_dir / f"queries_{split}.jsonl"
    row_count, query_to_doc_ids, qrels_doc_ids, errors = read_qrels(qrels_path)
    query_row_count, query_ids, query_errors = read_jsonl_ids(query_path, "query")
    errors.extend(query_errors)

    qrels_query_ids = set(query_to_doc_ids)
    qrels_query_ids_missing = sorted(qrels_query_ids - query_ids)
    qrels_doc_ids_missing = sorted(qrels_doc_ids - corpus_ids)
    if qrels_query_ids_missing:
        errors.append(
            f"{qrels_path.name} references {len(qrels_query_ids_missing)} query id(s) missing from {query_path.name}."
        )
    if qrels_doc_ids_missing:
        errors.append(
            f"{qrels_path.name} references {len(qrels_doc_ids_missing)} corpus id(s) missing from corpus.jsonl."
        )

    split_result: dict[str, Any] = {
        "qrels_file": qrels_path.name,
        "split": split,
        "qrels_rows": row_count,
        "unique_qrels_query_ids": len(qrels_query_ids),
        "unique_qrels_corpus_ids": len(qrels_doc_ids),
        **distribution_summary(query_to_doc_ids, show_top_queries),
        "query_file": query_path.name,
        "query_rows": query_row_count,
        "query_rows_with_at_least_one_qrel": len(query_ids & qrels_query_ids),
        "query_rows_with_no_qrels": len(query_ids - qrels_query_ids),
        "qrels_query_ids_missing_from_query_file": len(qrels_query_ids_missing),
        "missing_query_ids": qrels_query_ids_missing,
        "corpus_file": "corpus.jsonl",
        "corpus_rows": corpus_row_count,
        "qrels_document_ids_present_in_corpus": len(qrels_doc_ids & corpus_ids),
        "qrels_document_ids_missing_from_corpus": len(qrels_doc_ids_missing),
        "missing_corpus_ids": qrels_doc_ids_missing,
    }
    return split_result, errors


def analyze_qrels(config_path: Path, show_top_queries: int) -> tuple[dict[str, Any], list[str]]:
    config = load_dataset_config(config_path)
    if "output_dir" not in config:
        raise RuntimeError("Config must define output_dir.")
    output_dir = Path(str(config["output_dir"]))
    if not output_dir.is_dir():
        raise RuntimeError(f"Processed data directory does not exist: {output_dir}")

    qrels_files = [output_dir / f"qrels_{split}.tsv" for split in SPLITS if (output_dir / f"qrels_{split}.tsv").is_file()]
    if not qrels_files:
        raise RuntimeError(f"No qrels files found in {output_dir}. Expected qrels_train.tsv, qrels_dev.tsv, or qrels_test.tsv.")

    corpus_row_count, corpus_ids, corpus_errors = read_jsonl_ids(output_dir / "corpus.jsonl", "corpus")
    errors = list(corpus_errors)
    split_results: list[dict[str, Any]] = []
    for split in SPLITS:
        if not (output_dir / f"qrels_{split}.tsv").is_file():
            continue
        split_result, split_errors = analyze_split(output_dir, split, corpus_row_count, corpus_ids, show_top_queries)
        split_results.append(split_result)
        errors.extend(split_errors)

    return (
        {
            "config": str(config_path),
            "output_dir": str(output_dir),
            "corpus_file": "corpus.jsonl",
            "corpus_rows": corpus_row_count,
            "splits": split_results,
            "errors": errors,
        },
        errors,
    )


def print_terminal_summary(result: dict[str, Any]) -> None:
    print(f"Qrels analysis for: {result['config']}")
    print(f"Processed directory: {result['output_dir']}")
    print(f"Corpus rows: {result['corpus_rows']}")
    for split in result["splits"]:
        print("")
        print(f"{split['qrels_file']} ({split['split']}):")
        print(f"  qrels rows: {split['qrels_rows']}")
        print(f"  unique qrels query ids: {split['unique_qrels_query_ids']}")
        print(f"  unique qrels corpus ids: {split['unique_qrels_corpus_ids']}")
        print(
            "  relevant docs per qrels query: "
            f"min={split['min_relevant_docs_per_qrels_query']} "
            f"max={split['max_relevant_docs_per_qrels_query']} "
            f"mean={split['mean_relevant_docs_per_qrels_query']:.2f} "
            f"median={split['median_relevant_docs_per_qrels_query']:.2f}"
        )
        print(f"  queries with exactly 1 relevant document: {split['queries_with_exactly_1_relevant_doc']}")
        print(f"  queries with exactly 2 relevant documents: {split['queries_with_exactly_2_relevant_docs']}")
        print(f"  queries with exactly 3 relevant documents: {split['queries_with_exactly_3_relevant_docs']}")
        print(f"  queries with 4 or more relevant documents: {split['queries_with_4_or_more_relevant_docs']}")
        print(f"  query rows in {split['query_file']}: {split['query_rows']}")
        print(f"  query rows with at least one qrel: {split['query_rows_with_at_least_one_qrel']}")
        print(f"  query rows with no qrels: {split['query_rows_with_no_qrels']}")
        print(f"  qrels query ids missing from query file: {split['qrels_query_ids_missing_from_query_file']}")
        print(f"  qrels document ids present in corpus: {split['qrels_document_ids_present_in_corpus']}")
        print(f"  qrels document ids missing from corpus: {split['qrels_document_ids_missing_from_corpus']}")
        if split["top_queries"]:
            print("  top queries:")
            for query in split["top_queries"]:
                print(f"    {query['query_id']}: {query['relevant_docs']} relevant document(s)")
    if result["errors"]:
        print("", file=sys.stderr)
        print("Errors:", file=sys.stderr)
        for error in result["errors"]:
            print(f"  error: {error}", file=sys.stderr)
    else:
        print("")
        print("Qrels analysis passed.")


def write_json_output(path: Path, result: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv_output(path: Path, result: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    fieldnames = [
        "split",
        "qrels_file",
        "qrels_rows",
        "unique_qrels_query_ids",
        "unique_qrels_corpus_ids",
        "min_relevant_docs_per_qrels_query",
        "max_relevant_docs_per_qrels_query",
        "mean_relevant_docs_per_qrels_query",
        "median_relevant_docs_per_qrels_query",
        "queries_with_exactly_1_relevant_doc",
        "queries_with_exactly_2_relevant_docs",
        "queries_with_exactly_3_relevant_docs",
        "queries_with_4_or_more_relevant_docs",
        "query_file",
        "query_rows",
        "query_rows_with_at_least_one_qrel",
        "query_rows_with_no_qrels",
        "qrels_query_ids_missing_from_query_file",
        "corpus_rows",
        "qrels_document_ids_present_in_corpus",
        "qrels_document_ids_missing_from_corpus",
    ]
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split in result["splits"]:
            writer.writerow({field: split[field] for field in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a dataset YAML config.")
    parser.add_argument("--output-json", type=Path, help="Optional path for full JSON analysis.")
    parser.add_argument("--output-csv", type=Path, help="Optional path for compact per-split CSV summary.")
    parser.add_argument("--show-top-queries", type=int, default=10, help="Number of top queries to display and store.")
    args = parser.parse_args()

    if args.show_top_queries < 0:
        print("error: --show-top-queries must be non-negative.", file=sys.stderr)
        return 1

    try:
        result, errors = analyze_qrels(args.config, args.show_top_queries)
        print_terminal_summary(result)
        if args.output_json:
            write_json_output(args.output_json, result)
        if args.output_csv:
            write_csv_output(args.output_csv, result)
        return 0 if not errors else 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

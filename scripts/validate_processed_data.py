#!/usr/bin/env python3
"""Validate normalized retrieval files created by ``prepare_data.py``."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from data_utils import iter_jsonl, load_yaml, safe_text


QRELS_HEADER = ["query-id", "corpus-id", "score"]
TRIPLES_HEADER = ["query-id", "positive-doc-id", "negative-doc-id"]


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def add_missing_file(result: ValidationResult, path: Path, strict: bool) -> None:
    message = f"Missing optional file: {path.name}"
    if strict:
        result.errors.append(message)
    else:
        result.warnings.append(message)


def validate_jsonl_file(path: Path, kind: str, result: ValidationResult) -> set[str]:
    ids: set[str] = set()
    count = 0
    try:
        for line_number, row in enumerate(iter_jsonl(path), start=1):
            count += 1
            identifier = safe_text(row.get("_id"))
            if not identifier:
                result.errors.append(f"{path.name}:{line_number} has an empty _id.")
                continue
            if identifier in ids:
                result.errors.append(f"{path.name}:{line_number} duplicates _id '{identifier}'.")
            ids.add(identifier)
            if kind == "corpus":
                if "title" not in row:
                    result.errors.append(f"{path.name}:{line_number} is missing required field 'title'.")
                if not safe_text(row.get("text")):
                    result.errors.append(f"{path.name}:{line_number} has empty corpus text.")
            elif kind == "query" and not safe_text(row.get("text")):
                result.errors.append(f"{path.name}:{line_number} has empty query text.")
    except (OSError, ValueError) as exc:
        result.errors.append(str(exc))
    result.counts[path.name] = count
    return ids


def tsv_rows(path: Path, expected_header: list[str], result: ValidationResult) -> Iterable[tuple[int, list[str]]]:
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        result.errors.append(f"Could not open {path.name}: {exc}")
        return []

    with handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            result.errors.append(f"{path.name} is empty; expected header {expected_header}.")
            return []
        if header != expected_header:
            result.errors.append(f"{path.name} header must be {expected_header}; found {header}.")
        return [(line_number, row) for line_number, row in enumerate(reader, start=2) if row]


def validate_qrels(path: Path, query_ids: set[str] | None, corpus_ids: set[str], result: ValidationResult) -> None:
    rows = tsv_rows(path, QRELS_HEADER, result)
    count = 0
    for line_number, row in rows:
        count += 1
        if len(row) != 3:
            result.errors.append(f"{path.name}:{line_number} must contain exactly 3 columns.")
            continue
        query_id, corpus_id, score = (safe_text(value) for value in row)
        if query_ids is not None and query_id not in query_ids:
            result.errors.append(f"{path.name}:{line_number} references missing query id '{query_id}'.")
        if corpus_id not in corpus_ids:
            result.errors.append(f"{path.name}:{line_number} references missing corpus id '{corpus_id}'.")
        try:
            int(score)
        except ValueError:
            result.errors.append(f"{path.name}:{line_number} score '{score}' is not integer-like.")
    result.counts[path.name] = count


def validate_triples(path: Path, query_ids: set[str] | None, corpus_ids: set[str], result: ValidationResult) -> None:
    rows = tsv_rows(path, TRIPLES_HEADER, result)
    count = 0
    for line_number, row in rows:
        count += 1
        if len(row) != 3:
            result.errors.append(f"{path.name}:{line_number} must contain exactly 3 columns.")
            continue
        query_id, positive_id, negative_id = (safe_text(value) for value in row)
        if query_ids is None or query_id not in query_ids:
            result.errors.append(f"{path.name}:{line_number} references missing train query id '{query_id}'.")
        if positive_id not in corpus_ids:
            result.errors.append(f"{path.name}:{line_number} references missing positive corpus id '{positive_id}'.")
        if negative_id not in corpus_ids:
            result.errors.append(f"{path.name}:{line_number} references missing negative corpus id '{negative_id}'.")
        if positive_id == negative_id:
            result.errors.append(f"{path.name}:{line_number} uses the same positive and negative document id.")
    result.counts[path.name] = count


def required_files_for_task(task: str) -> tuple[str, ...]:
    if task == "train_and_eval":
        return ("corpus.jsonl", "queries_train.jsonl", "qrels_train.tsv", "triples_train.tsv")
    if task == "ood_eval":
        return ("corpus.jsonl",)
    return ()


def validate_processed_data(config: Mapping[str, Any], strict: bool = False) -> ValidationResult:
    """Validate a normalized dataset directory without requiring external packages."""
    result = ValidationResult()
    output_dir = Path(config["output_dir"])
    if not output_dir.is_dir():
        result.errors.append(f"Processed data directory does not exist: {output_dir}")
        return result

    task = safe_text(config.get("task"))
    for filename in required_files_for_task(task):
        path = output_dir / filename
        if not path.is_file():
            add_missing_file(result, path, strict)

    corpus_path = output_dir / "corpus.jsonl"
    corpus_ids: set[str] = set()
    if corpus_path.is_file():
        corpus_ids = validate_jsonl_file(corpus_path, "corpus", result)

    query_ids_by_role: dict[str, set[str]] = {}
    for path in sorted(output_dir.glob("queries_*.jsonl")):
        role = path.stem.removeprefix("queries_")
        query_ids_by_role[role] = validate_jsonl_file(path, "query", result)

    query_files = list(output_dir.glob("queries_*.jsonl"))
    qrels_files = list(output_dir.glob("qrels_*.tsv"))
    if task == "ood_eval" and strict:
        if not query_files:
            result.errors.append("OOD dataset requires at least one queries_*.jsonl file in strict mode.")
        if not qrels_files:
            result.errors.append("OOD dataset requires at least one qrels_*.tsv file in strict mode.")
    elif task == "ood_eval":
        if not query_files:
            result.warnings.append("OOD dataset has no queries_*.jsonl files.")
        if not qrels_files:
            result.warnings.append("OOD dataset has no qrels_*.tsv files.")

    for path in sorted(qrels_files):
        role = path.stem.removeprefix("qrels_")
        validate_qrels(path, query_ids_by_role.get(role), corpus_ids, result)

    triples_path = output_dir / "triples_train.tsv"
    if triples_path.is_file():
        validate_triples(triples_path, query_ids_by_role.get("train"), corpus_ids, result)

    return result


def print_result(result: ValidationResult) -> None:
    for filename, count in sorted(result.counts.items()):
        print(f"{filename}: {count} rows")
    for warning in result.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for error in result.errors:
        print(f"error: {error}", file=sys.stderr)
    if result.ok:
        print("Processed data validation passed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a dataset YAML config.")
    parser.add_argument("--strict", action="store_true", help="Require files appropriate for the configured dataset task.")
    args = parser.parse_args()
    try:
        config = load_yaml(args.config)
        if "output_dir" not in config or "task" not in config:
            raise RuntimeError("Config must define output_dir and task.")
        result = validate_processed_data(config, strict=args.strict)
        print_result(result)
        return 0 if result.ok else 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

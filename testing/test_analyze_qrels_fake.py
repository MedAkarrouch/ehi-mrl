#!/usr/bin/env python3
"""Offline subprocess tests for qrels diagnostics using fake processed data."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_tsv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_config(path: Path, output_dir: Path) -> None:
    path.write_text(f"dataset_name: fake\noutput_dir: {output_dir}\ntask: train_and_eval\n", encoding="utf-8")


def create_valid_processed_dataset(directory: Path) -> None:
    write_jsonl(
        directory / "corpus.jsonl",
        [
            {"_id": "d1", "title": "", "text": "doc one"},
            {"_id": "d2", "title": "", "text": "doc two"},
            {"_id": "d3", "title": "", "text": "doc three"},
            {"_id": "d4", "title": "", "text": "doc four"},
            {"_id": "d5", "title": "", "text": "doc five"},
            {"_id": "d6", "title": "", "text": "doc six"},
        ],
    )
    write_jsonl(
        directory / "queries_train.jsonl",
        [
            {"_id": "q1", "text": "query one"},
            {"_id": "q2", "text": "query two"},
            {"_id": "q3", "text": "query three"},
            {"_id": "q4", "text": "query with no qrels"},
        ],
    )
    write_tsv(
        directory / "qrels_train.tsv",
        ("query-id", "corpus-id", "score"),
        [
            ("q1", "d1", "1"),
            ("q2", "d2", "1"),
            ("q2", "d3", "1"),
            ("q3", "d4", "1"),
            ("q3", "d5", "1"),
            ("q3", "d6", "1"),
        ],
    )


def create_invalid_processed_dataset(directory: Path) -> None:
    write_jsonl(directory / "corpus.jsonl", [{"_id": "d1", "title": "", "text": "doc one"}])
    write_jsonl(directory / "queries_train.jsonl", [{"_id": "q1", "text": "query one"}])
    write_tsv(
        directory / "qrels_train.tsv",
        ("query-id", "corpus-id", "score"),
        [("q1", "missing-doc", "1")],
    )


def run_analyzer(config_path: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(ROOT / "scripts" / "analyze_qrels.py"), "--config", str(config_path), *extra_args]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def test_valid_dataset(root: Path) -> None:
    processed = root / "valid"
    config_path = root / "valid.yaml"
    json_path = processed / "qrels_analysis.json"
    csv_path = processed / "qrels_analysis.csv"
    create_valid_processed_dataset(processed)
    write_config(config_path, processed)

    completed = run_analyzer(config_path, "--output-json", str(json_path), "--output-csv", str(csv_path))
    assert completed.returncode == 0, completed.stderr
    assert "qrels rows: 6" in completed.stdout
    assert "unique qrels query ids: 3" in completed.stdout
    assert "queries with exactly 1 relevant document: 1" in completed.stdout
    assert "queries with exactly 2 relevant documents: 1" in completed.stdout
    assert "queries with exactly 3 relevant documents: 1" in completed.stdout
    assert "query rows with no qrels: 1" in completed.stdout

    analysis = json.loads(json_path.read_text(encoding="utf-8"))
    split = analysis["splits"][0]
    assert split["split"] == "train"
    assert split["qrels_rows"] == 6
    assert split["unique_qrels_query_ids"] == 3
    assert split["queries_with_exactly_1_relevant_doc"] == 1
    assert split["queries_with_exactly_2_relevant_docs"] == 1
    assert split["queries_with_exactly_3_relevant_docs"] == 1
    assert split["query_rows_with_no_qrels"] == 1
    assert split["qrels_document_ids_missing_from_corpus"] == 0

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["split"] == "train"
    assert rows[0]["qrels_rows"] == "6"
    assert rows[0]["query_rows_with_no_qrels"] == "1"


def test_missing_corpus_id_fails(root: Path) -> None:
    processed = root / "invalid"
    config_path = root / "invalid.yaml"
    create_invalid_processed_dataset(processed)
    write_config(config_path, processed)

    completed = run_analyzer(config_path)
    assert completed.returncode != 0
    combined_output = completed.stdout + completed.stderr
    assert "missing from corpus" in combined_output


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        test_valid_dataset(root)
        test_missing_corpus_id_fails(root)
    print("Qrels analyzer fake-data checks passed.")


if __name__ == "__main__":
    main()

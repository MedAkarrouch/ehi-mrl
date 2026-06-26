#!/usr/bin/env python3
"""Fake-data checks for evaluator metric selection without Hit@1."""

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


def run_eval(config_path: Path, json_path: Path, csv_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_run.py"),
            "--config",
            str(config_path),
            "--output-json",
            str(json_path),
            "--output-csv",
            str(csv_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        processed = root / "processed"
        results = root / "results"
        write_jsonl(processed / "queries_test.jsonl", [{"_id": "q1", "text": "one"}, {"_id": "q2", "text": "two"}])
        write_tsv(processed / "qrels_test.tsv", ("query-id", "corpus-id", "score"), [("q1", "d1", "1"), ("q2", "d2", "1")])
        write_tsv(
            results / "run_test.tsv",
            ("query-id", "corpus-id", "score", "rank"),
            [("q1", "d1", "1.0", "1"), ("q1", "d2", "0.5", "2"), ("q2", "d1", "1.0", "1"), ("q2", "d2", "0.5", "2")],
        )
        dataset_config = root / "data.yaml"
        dataset_config.write_text(f"dataset_name: fake\noutput_dir: {processed}\n", encoding="utf-8")
        selected_config = root / "selected.yaml"
        selected_config.write_text(
            "\n".join(
                [
                    f"dataset_config: {dataset_config}",
                    "dataset_name: fake",
                    "split: test",
                    "query_file: queries_test.jsonl",
                    "qrels_file: qrels_test.tsv",
                    f"results_dir: {results}",
                    "metrics:",
                    "  - MRR@10",
                    "  - Recall@10",
                    "  - Recall@100",
                    "  - nDCG@10",
                    "primary_metrics:",
                    "  - MRR@10",
                    "  - Recall@100",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        selected_json = root / "selected.json"
        selected_csv = root / "selected.csv"
        completed = run_eval(selected_config, selected_json, selected_csv)
        assert completed.returncode == 0, completed.stderr
        assert "Hit@1" not in selected_json.read_text(encoding="utf-8")
        assert "Hit@1" not in selected_csv.read_text(encoding="utf-8")

        default_config = root / "default.yaml"
        default_config.write_text(
            "\n".join(
                [
                    f"dataset_config: {dataset_config}",
                    "dataset_name: fake",
                    "split: test",
                    "query_file: queries_test.jsonl",
                    "qrels_file: qrels_test.tsv",
                    f"results_dir: {results}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        default_json = root / "default.json"
        default_csv = root / "default.csv"
        completed = run_eval(default_config, default_json, default_csv)
        assert completed.returncode == 0, completed.stderr
        assert "Hit@1" in default_json.read_text(encoding="utf-8")
        assert "Hit@1" in default_csv.read_text(encoding="utf-8")
    print("Evaluator metric-selection checks passed.")


if __name__ == "__main__":
    main()

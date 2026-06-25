#!/usr/bin/env python3
"""Offline subprocess tests for retrieval evaluation metrics."""

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


def write_configs(root: Path, processed_dir: Path, results_dir: Path) -> Path:
    dataset_config = root / "data.yaml"
    baseline_config = root / "exact.yaml"
    encoder_config = root / "encoder.yaml"
    dataset_config.write_text(f"dataset_name: fake\noutput_dir: {processed_dir}\ntask: ood_eval\n", encoding="utf-8")
    encoder_config.write_text(
        "encoder_name: fake\nhf_model_name: fake\nnormalize_embeddings: true\n",
        encoding="utf-8",
    )
    baseline_config.write_text(
        "\n".join(
            [
                f"dataset_config: {dataset_config}",
                f"encoder_config: {encoder_config}",
                "dataset_name: fake",
                "split: test",
                "query_file: queries_test.jsonl",
                "qrels_file: qrels_test.tsv",
                "top_k: 100",
                "similarity: cosine",
                f"embedding_dir: {root / 'embeddings'}",
                f"results_dir: {results_dir}",
                "device: cpu",
                "query_batch_size: 2",
                "corpus_chunk_size: 2",
                "primary_metrics:",
                "  - Hit@1",
                "  - MRR@10",
                "  - Recall@10",
                "  - nDCG@10",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return baseline_config


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        processed_dir = root / "processed"
        results_dir = root / "results"
        baseline_config = write_configs(root, processed_dir, results_dir)

        write_jsonl(
            processed_dir / "corpus.jsonl",
            [
                {"_id": "d1", "title": "", "text": "doc one"},
                {"_id": "d2", "title": "", "text": "doc two"},
                {"_id": "d3", "title": "", "text": "doc three"},
                {"_id": "d4", "title": "", "text": "doc four"},
            ],
        )
        write_jsonl(
            processed_dir / "queries_test.jsonl",
            [
                {"_id": "q1", "text": "query one"},
                {"_id": "q2", "text": "query two"},
                {"_id": "q3", "text": "query without qrels"},
            ],
        )
        write_tsv(
            processed_dir / "qrels_test.tsv",
            ("query-id", "corpus-id", "score"),
            [("q1", "d1", "1"), ("q2", "d2", "1"), ("q2", "d3", "1")],
        )
        write_tsv(
            results_dir / "run_test.tsv",
            ("query-id", "corpus-id", "score", "rank"),
            [
                ("q1", "d1", "1.0", "1"),
                ("q1", "d4", "0.1", "2"),
                ("q2", "d4", "0.9", "1"),
                ("q2", "d2", "0.8", "2"),
                ("q2", "d3", "0.7", "3"),
                ("q3", "d3", "1.0", "1"),
            ],
        )

        json_path = results_dir / "metrics.json"
        csv_path = results_dir / "metrics.csv"
        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_run.py"),
                "--config",
                str(baseline_config),
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
        assert completed.returncode == 0, completed.stderr
        assert "query rows with no qrels: 1" in completed.stdout
        result = json.loads(json_path.read_text(encoding="utf-8"))
        assert result["diagnostics"]["evaluated_queries"] == 2
        assert result["diagnostics"]["query_rows"] == 3
        assert result["diagnostics"]["query_rows_with_no_qrels"] == 1
        assert abs(result["metrics"]["Hit@1"] - 0.5) < 1e-9
        assert abs(result["metrics"]["MRR@10"] - 0.75) < 1e-9
        assert abs(result["metrics"]["Recall@10"] - 1.0) < 1e-9
        assert "nDCG@10" in result["metrics"]

        with csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["evaluated_queries"] == "2"
        assert abs(float(rows[0]["Hit@1"]) - 0.5) < 1e-9
    print("Retrieval metric fake-data checks passed.")


if __name__ == "__main__":
    main()

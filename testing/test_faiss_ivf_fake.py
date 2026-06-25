#!/usr/bin/env python3
"""Tiny fake-data subprocess test for CPU FAISS IVF scripts."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def optional_faiss_and_numpy():
    try:
        import faiss  # noqa: F401
        import numpy as np
    except ImportError:
        print("FAISS IVF fake-data checks skipped: faiss or numpy is not installed locally.")
        return None
    return np


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


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def main() -> None:
    np = optional_faiss_and_numpy()
    if np is None:
        return
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        processed_dir = root / "processed"
        embedding_dir = root / "embeddings"
        index_dir = root / "indexes"
        results_dir = root / "results"
        embedding_dir.mkdir(parents=True)

        corpus_embeddings = np.asarray(
            [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [0.05, 0.95]],
            dtype=np.float32,
        )
        query_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        np.save(embedding_dir / "corpus_embeddings.npy", corpus_embeddings)
        np.save(embedding_dir / "queries_test_embeddings.npy", query_embeddings)
        (embedding_dir / "corpus_ids.json").write_text(json.dumps(["d1", "d2", "d3", "d4"]), encoding="utf-8")
        (embedding_dir / "queries_test_ids.json").write_text(json.dumps(["q1", "q2"]), encoding="utf-8")

        write_jsonl(processed_dir / "corpus.jsonl", [{"_id": f"d{i}", "title": "", "text": f"doc {i}"} for i in range(1, 5)])
        write_jsonl(processed_dir / "queries_test.jsonl", [{"_id": "q1", "text": "one"}, {"_id": "q2", "text": "two"}])
        write_tsv(processed_dir / "qrels_test.tsv", ("query-id", "corpus-id", "score"), [("q1", "d1", "1"), ("q2", "d3", "1")])

        dataset_config = root / "data.yaml"
        exact_config = root / "exact.yaml"
        faiss_config = root / "faiss.yaml"
        dataset_config.write_text(f"dataset_name: fake\noutput_dir: {processed_dir}\ntask: ood_eval\n", encoding="utf-8")
        exact_config.write_text(
            "\n".join(
                [
                    f"dataset_config: {dataset_config}",
                    "dataset_name: fake",
                    "split: test",
                    "query_file: queries_test.jsonl",
                    "qrels_file: qrels_test.tsv",
                    "top_k: 3",
                    "similarity: cosine",
                    f"embedding_dir: {embedding_dir}",
                    f"results_dir: {root / 'exact_results'}",
                    "primary_metrics:",
                    "  - Hit@1",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        faiss_config.write_text(
            "\n".join(
                [
                    f"exact_baseline_config: {exact_config}",
                    "dataset_name: fake",
                    "split: test",
                    f"embedding_dir: {embedding_dir}",
                    f"index_dir: {index_dir}",
                    f"results_dir: {results_dir}",
                    "metric: inner_product",
                    "top_k: 3",
                    "nlist_values: [2]",
                    "nprobe_values: [1, 2]",
                    "max_train_vectors: null",
                    "omp_threads: 1",
                    "primary_metrics:",
                    "  - Hit@1",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        completed = run([sys.executable, str(ROOT / "scripts" / "build_faiss_ivf.py"), "--config", str(faiss_config), "--nlist", "2"])
        assert completed.returncode == 0, completed.stderr
        assert (index_dir / "ivf_nlist2.faiss").is_file()

        completed = run([sys.executable, str(ROOT / "scripts" / "search_faiss_ivf.py"), "--config", str(faiss_config), "--nlist", "2", "--nprobe", "2"])
        assert completed.returncode == 0, completed.stderr
        run_file = results_dir / "run_test_nlist2_nprobe2.tsv"
        search_info = results_dir / "search_info_test_nlist2_nprobe2.json"
        assert run_file.is_file()
        assert search_info.is_file()

        metrics_json = results_dir / "metrics.json"
        completed = run(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_run.py"),
                "--config",
                str(exact_config),
                "--run-file",
                str(run_file),
                "--output-json",
                str(metrics_json),
                "--output-csv",
                str(results_dir / "metrics.csv"),
            ]
        )
        assert completed.returncode == 0, completed.stderr
        assert metrics_json.is_file()
        info = json.loads(search_info.read_text(encoding="utf-8"))
        assert "percent_docs_visited" in info
        assert "latency_ms_per_query" in info
    print("FAISS IVF fake-data checks passed.")


if __name__ == "__main__":
    main()

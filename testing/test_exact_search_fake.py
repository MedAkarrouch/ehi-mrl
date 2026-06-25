#!/usr/bin/env python3
"""Offline subprocess tests for exact search with tiny embeddings."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_config(path: Path, embedding_dir: Path, results_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "dataset_name: fake",
                "split: test",
                "top_k: 2",
                "similarity: cosine",
                f"embedding_dir: {embedding_dir}",
                f"results_dir: {results_dir}",
                "device: cpu",
                "query_batch_size: 1",
                "corpus_chunk_size: 2",
                "primary_metrics:",
                "  - Hit@1",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_npy_float32(path: Path, rows: list[list[float]]) -> None:
    row_count = len(rows)
    column_count = len(rows[0]) if rows else 0
    header = {
        "descr": "<f4",
        "fortran_order": False,
        "shape": (row_count, column_count),
    }
    header_text = repr(header)
    padding = " " * ((16 - ((10 + len(header_text) + 1) % 16)) % 16)
    header_bytes = f"{header_text}{padding}\n".encode("latin1")
    values = [value for row in rows for value in row]
    with path.open("wb") as handle:
        handle.write(b"\x93NUMPY")
        handle.write(bytes([1, 0]))
        handle.write(struct.pack("<H", len(header_bytes)))
        handle.write(header_bytes)
        handle.write(struct.pack("<" + "f" * len(values), *values))


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        embedding_dir = root / "embeddings"
        results_dir = root / "results"
        embedding_dir.mkdir(parents=True)
        config_path = root / "exact.yaml"
        write_config(config_path, embedding_dir, results_dir)

        write_npy_float32(embedding_dir / "corpus_embeddings.npy", [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        write_npy_float32(embedding_dir / "queries_test_embeddings.npy", [[1.0, 0.0], [0.0, 1.0]])
        (embedding_dir / "corpus_ids.json").write_text(json.dumps(["c1", "c2", "c3"]), encoding="utf-8")
        (embedding_dir / "queries_test_ids.json").write_text(json.dumps(["q1", "q2"]), encoding="utf-8")

        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "exact_search.py"), "--config", str(config_path), "--overwrite"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        run_path = results_dir / "run_test.tsv"
        assert run_path.is_file()
        lines = run_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == "query-id\tcorpus-id\tscore\trank"
        rows = [line.split("\t") for line in lines[1:]]
        assert rows[0][0] == "q1"
        assert rows[0][1] == "c1"
        assert rows[0][3] == "1"
        q2_rows = [row for row in rows if row[0] == "q2"]
        assert q2_rows[0][1] == "c2"
        assert q2_rows[0][3] == "1"
    print("Exact-search fake embedding checks passed.")


if __name__ == "__main__":
    main()

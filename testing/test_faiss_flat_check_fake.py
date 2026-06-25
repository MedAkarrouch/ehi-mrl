#!/usr/bin/env python3
"""Tiny fake-data subprocess test for FAISS flat exact-check script."""

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
        print("FAISS flat-check fake-data test skipped: faiss or numpy is not installed locally.")
        return None
    return np


def write_tsv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    np = optional_faiss_and_numpy()
    if np is None:
        return
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        embedding_dir = root / "embeddings"
        results_dir = root / "results"
        embedding_dir.mkdir(parents=True)
        results_dir.mkdir(parents=True)
        np.save(embedding_dir / "corpus_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        np.save(embedding_dir / "queries_test_embeddings.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        (embedding_dir / "corpus_ids.json").write_text(json.dumps(["d1", "d2"]), encoding="utf-8")
        (embedding_dir / "queries_test_ids.json").write_text(json.dumps(["q1", "q2"]), encoding="utf-8")
        write_tsv(
            results_dir / "run_test.tsv",
            ("query-id", "corpus-id", "score", "rank"),
            [("q1", "d1", "1.0", "1"), ("q1", "d2", "0.0", "2"), ("q2", "d2", "1.0", "1"), ("q2", "d1", "0.0", "2")],
        )
        exact_config = root / "exact.yaml"
        exact_config.write_text(
            "\n".join(
                [
                    "dataset_name: fake",
                    "split: test",
                    "top_k: 2",
                    f"embedding_dir: {embedding_dir}",
                    f"results_dir: {results_dir}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "faiss_flat_check.py"), "--exact-config", str(exact_config)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "top1 agreement" in completed.stdout
    print("FAISS flat-check fake-data checks passed.")


if __name__ == "__main__":
    main()

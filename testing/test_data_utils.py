#!/usr/bin/env python3
"""Offline unit tests for normalized data-file helpers."""

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_utils import count_jsonl, count_tsv_rows, read_jsonl, read_tsv, safe_text, write_jsonl, write_tsv  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        jsonl_path = root / "rows.jsonl"
        tsv_path = root / "rows.tsv"
        json_rows = [{"_id": "d1", "text": "first"}, {"_id": "d2", "text": "second"}]
        tsv_rows = [("q1", "d1", 1), ("q2", "d2", 0)]

        write_jsonl(jsonl_path, json_rows)
        write_tsv(tsv_path, ("query-id", "corpus-id", "score"), tsv_rows)

        assert read_jsonl(jsonl_path) == json_rows
        assert read_tsv(tsv_path) == (["query-id", "corpus-id", "score"], [["q1", "d1", "1"], ["q2", "d2", "0"]])
        assert count_jsonl(jsonl_path) == 2
        assert count_tsv_rows(tsv_path) == 2
        assert safe_text(None) == ""
        assert safe_text("  normalized text  ") == "normalized text"
        assert safe_text(42) == "42"
    print("Data utility checks passed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline validation tests using tiny synthetic processed retrieval data."""

import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from data_utils import write_jsonl, write_tsv  # noqa: E402
from validate_processed_data import validate_processed_data  # noqa: E402


def create_dataset(directory: Path, invalid_qrels: bool = False) -> None:
    write_jsonl(
        directory / "corpus.jsonl",
        [
            {"_id": "d1", "title": "Document one", "text": "First document."},
            {"_id": "d2", "title": "Document two", "text": "Second document."},
        ],
    )
    write_jsonl(directory / "queries_train.jsonl", [{"_id": "q1", "text": "A test query"}])
    qrels_doc_id = "missing-doc" if invalid_qrels else "d1"
    write_tsv(directory / "qrels_train.tsv", ("query-id", "corpus-id", "score"), [("q1", qrels_doc_id, 1)])
    write_tsv(
        directory / "triples_train.tsv",
        ("query-id", "positive-doc-id", "negative-doc-id"),
        [("q1", "d1", "d2")],
    )


def create_ood_dataset(directory: Path, invalid_qrels: bool = False) -> None:
    write_jsonl(
        directory / "corpus.jsonl",
        [
            {"_id": "d1", "title": "Document one", "text": "First document."},
            {"_id": "d2", "title": "Document two", "text": "Second document."},
        ],
    )
    write_jsonl(directory / "queries_test.jsonl", [{"_id": "q1", "text": "An OOD test query"}])
    qrels_doc_id = "missing-doc" if invalid_qrels else "d1"
    write_tsv(directory / "qrels_test.tsv", ("query-id", "corpus-id", "score"), [("q1", qrels_doc_id, 1)])


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        config = {"output_dir": str(root / "valid"), "task": "train_and_eval"}
        create_dataset(Path(config["output_dir"]))
        valid_result = validate_processed_data(config, strict=True)
        assert valid_result.ok, valid_result.errors

        invalid_config = {"output_dir": str(root / "invalid"), "task": "train_and_eval"}
        create_dataset(Path(invalid_config["output_dir"]), invalid_qrels=True)
        invalid_result = validate_processed_data(invalid_config, strict=True)
        assert not invalid_result.ok
        assert any("missing corpus id" in error for error in invalid_result.errors), invalid_result.errors

        ood_config = {"output_dir": str(root / "ood-valid"), "task": "ood_eval"}
        create_ood_dataset(Path(ood_config["output_dir"]))
        ood_result = validate_processed_data(ood_config, strict=True)
        assert ood_result.ok, ood_result.errors

        ood_invalid_config = {"output_dir": str(root / "ood-invalid"), "task": "ood_eval"}
        create_ood_dataset(Path(ood_invalid_config["output_dir"]), invalid_qrels=True)
        ood_invalid_result = validate_processed_data(ood_invalid_config, strict=True)
        assert not ood_invalid_result.ok
        assert any("missing corpus id" in error for error in ood_invalid_result.errors), ood_invalid_result.errors
    print("Processed-data validator fixture checks passed.")


if __name__ == "__main__":
    main()

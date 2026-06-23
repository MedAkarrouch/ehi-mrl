#!/usr/bin/env python3
"""Offline checks for the explicit BEIR corpus and queries configurations."""

import re
import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_data import prepare_beir  # noqa: E402
FILES = ("data_beir_scifact.yaml", "data_beir_fiqa.yaml")
REQUIRED = (
    "dataset_name",
    "source",
    "hf_dataset",
    "hf_corpus_config",
    "hf_queries_config",
    "hf_corpus_split",
    "hf_queries_split",
    "hf_qrels",
    "task",
    "output_dir",
    "cache_dir",
)
EXPECTED = {
    "hf_corpus_config": "corpus",
    "hf_queries_config": "queries",
    "hf_corpus_split": "corpus",
    "hf_queries_split": "queries",
}


def value_for(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    return match.group(1) if match else None


def main() -> None:
    for filename in FILES:
        path = ROOT / "configs" / filename
        assert path.is_file(), f"Missing BEIR config: {path}"
        text = path.read_text(encoding="utf-8")
        for key in REQUIRED:
            assert value_for(text, key), f"{filename} is missing required key: {key}"
        for key, expected_value in EXPECTED.items():
            assert value_for(text, key) == expected_value, f"{filename} must set {key}: {expected_value}"

    calls: list[tuple[str, str | None, str | None]] = []

    def fake_load_dataset(dataset_id: str, config_name: str | None = None, **kwargs: str) -> list[dict[str, str | int]]:
        calls.append((dataset_id, config_name, kwargs.get("split")))
        if dataset_id == "BeIR/example" and config_name == "corpus":
            return [{"_id": "d1", "title": "Example", "text": "Corpus text"}]
        if dataset_id == "BeIR/example" and config_name == "queries":
            return [{"_id": "q1", "title": "Query title fallback"}]
        if dataset_id == "BeIR/example-qrels" and kwargs.get("split") == "test":
            return [{"query-id": "q1", "corpus-id": "d1", "score": 1}]
        raise AssertionError(f"Unexpected Hugging Face call: {dataset_id}, {config_name}, {kwargs}")

    config = {
        "hf_dataset": "BeIR/example",
        "hf_corpus_config": "corpus",
        "hf_queries_config": "queries",
        "hf_corpus_split": "corpus",
        "hf_queries_split": "queries",
        "hf_qrels": "BeIR/example-qrels",
    }
    with patch("prepare_data.load_dataset", side_effect=fake_load_dataset), patch(
        "prepare_data.available_hf_splits", return_value=["train", "test"]
    ):
        outputs = prepare_beir(config, Path(".cache/test"), max_docs=None, max_queries=None)
    assert calls == [
        ("BeIR/example", "corpus", "corpus"),
        ("BeIR/example", "queries", "queries"),
        ("BeIR/example-qrels", None, "test"),
    ], calls
    assert set(outputs) == {"corpus.jsonl", "queries_test.jsonl", "qrels_test.tsv"}
    assert outputs["queries_test.jsonl"][1] == [{"_id": "q1", "text": "Query title fallback"}]
    print("BEIR configuration loading checks passed.")


if __name__ == "__main__":
    main()

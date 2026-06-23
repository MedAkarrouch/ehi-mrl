#!/usr/bin/env python3
"""Normalize NQ320K and BEIR datasets into a common retrieval-data format."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from data_utils import count_jsonl, count_tsv_rows, ensure_dir, load_yaml, safe_text, set_hf_cache, write_jsonl, write_tsv


JSONL_FILENAMES = ("corpus.jsonl", "queries_train.jsonl", "queries_dev.jsonl", "queries_test.jsonl")
TSV_FILENAMES = ("qrels_train.tsv", "qrels_dev.tsv", "qrels_test.tsv", "triples_train.tsv")
NORMALIZED_FILENAMES = JSONL_FILENAMES + TSV_FILENAMES + ("dataset_info.json",)
QRELS_HEADER = ("query-id", "corpus-id", "score")
TRIPLES_HEADER = ("query-id", "positive-doc-id", "negative-doc-id")


def require_keys(config: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise RuntimeError(f"Config is missing required keys: {', '.join(missing)}")


def load_dataset(*args: Any, **kwargs: Any) -> Any:
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required; install project dependencies first.") from exc
    return hf_load_dataset(*args, **kwargs)


def first_value(row: Mapping[str, Any], *names: str) -> Any:
    normalized = {str(key).lower().replace("_", "-"): value for key, value in row.items()}
    for name in names:
        key = name.lower().replace("_", "-")
        if key in normalized:
            return normalized[key]
    return None


def limit_reached(count: int, limit: int | None) -> bool:
    return limit is not None and count >= limit


def clean_normalized_outputs(output_dir: Path, overwrite: bool) -> None:
    existing = [output_dir / filename for filename in NORMALIZED_FILENAMES if (output_dir / filename).exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise RuntimeError(
            f"Output directory '{output_dir}' already contains normalized files ({names}). "
            "Pass --overwrite to replace them."
        )
    if overwrite:
        for path in existing:
            path.unlink()


def choose_negative(doc_ids: list[str], positive_id: str, rng: random.Random) -> str:
    if len(doc_ids) < 2:
        raise RuntimeError("At least two corpus documents are required to generate NQ320K training negatives.")
    negative_id = rng.choice(doc_ids)
    while negative_id == positive_id:
        negative_id = rng.choice(doc_ids)
    return negative_id


def write_nq_corpus(source: Iterable[Mapping[str, Any]], output_path: Path, max_docs: int | None) -> list[str]:
    """Stream the large NQ corpus to disk while retaining only document identifiers."""
    ensure_dir(output_path.parent)
    doc_ids: list[str] = []
    seen_ids: set[str] = set()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in source:
            if limit_reached(len(doc_ids), max_docs):
                break
            raw_id = safe_text(first_value(row, "docid", "_id", "id"))
            text = safe_text(first_value(row, "document", "text", "contents", "body"))
            if not raw_id or not text:
                continue
            doc_id = f"NQDOC_{raw_id}"
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            normalized_row = {"_id": doc_id, "title": safe_text(first_value(row, "title")), "text": text}
            handle.write(json.dumps(normalized_row, ensure_ascii=False) + "\n")
            doc_ids.append(doc_id)
    if not doc_ids:
        raise RuntimeError("NQ320K corpus normalization produced no valid documents.")
    return doc_ids


def normalize_nq_pairs(
    source: Iterable[Mapping[str, Any]],
    prefix: str,
    corpus_ids: set[str],
    doc_ids: list[str],
    rng: random.Random,
    max_queries: int | None,
    max_pairs: int | None,
    include_negatives: bool,
) -> tuple[list[dict[str, str]], list[tuple[str, str, int]], list[tuple[str, str, str]]]:
    queries: list[dict[str, str]] = []
    qrels: list[tuple[str, str, int]] = []
    triples: list[tuple[str, str, str]] = []
    inspected_pairs = 0
    for row_index, row in enumerate(source):
        if limit_reached(inspected_pairs, max_pairs) or limit_reached(len(queries), max_queries):
            break
        inspected_pairs += 1
        query_text = safe_text(first_value(row, "query", "text", "question"))
        raw_doc_id = safe_text(first_value(row, "docid", "corpus-id", "document-id"))
        positive_id = f"NQDOC_{raw_doc_id}" if raw_doc_id else ""
        if not query_text or positive_id not in corpus_ids:
            continue
        query_id = f"NQ{prefix}_{row_index}"
        queries.append({"_id": query_id, "text": query_text})
        qrels.append((query_id, positive_id, 1))
        if include_negatives:
            triples.append((query_id, positive_id, choose_negative(doc_ids, positive_id, rng)))
    return queries, qrels, triples


def optional_nq_dev_dataset(config: Mapping[str, Any], cache_dir: Path) -> Any | None:
    try:
        return load_dataset(
            config["hf_dataset"],
            config.get("hf_pairs_config", "pairs"),
            split=config.get("hf_dev_split", "validation"),
            cache_dir=str(cache_dir),
        )
    except Exception as exc:  # Dataset revisions may not expose a validation split.
        print(f"warning: NQ320K validation split is unavailable and will be skipped: {exc}", file=sys.stderr)
        return None


def prepare_nq320k(
    config: Mapping[str, Any], output_dir: Path, cache_dir: Path, max_docs: int | None, max_queries: int | None, max_pairs: int | None
) -> dict[str, tuple[str, Any]]:
    require_keys(config, ("hf_corpus_config", "hf_pairs_config", "hf_corpus_split", "hf_train_split"))
    corpus_source = load_dataset(
        config["hf_dataset"],
        config["hf_corpus_config"],
        split=config["hf_corpus_split"],
        cache_dir=str(cache_dir),
    )
    doc_ids = write_nq_corpus(corpus_source, output_dir / "corpus.jsonl", max_docs)
    corpus_id_set = set(doc_ids)
    rng = random.Random(int(config.get("random_seed", 42)))

    train_source = load_dataset(
        config["hf_dataset"],
        config["hf_pairs_config"],
        split=config["hf_train_split"],
        cache_dir=str(cache_dir),
    )
    train_queries, train_qrels, triples = normalize_nq_pairs(
        train_source, "TRAIN", corpus_id_set, doc_ids, rng, max_queries, max_pairs, include_negatives=True
    )
    if not train_queries:
        raise RuntimeError("NQ320K train normalization produced no valid query/document pairs.")

    outputs: dict[str, tuple[str, Any]] = {
        "corpus.jsonl": ("existing_jsonl", None),
        "queries_train.jsonl": ("jsonl", train_queries),
        "qrels_train.tsv": ("qrels", train_qrels),
        "triples_train.tsv": ("triples", triples),
    }
    dev_source = optional_nq_dev_dataset(config, cache_dir)
    if dev_source is not None:
        dev_queries, dev_qrels, _triples = normalize_nq_pairs(
            dev_source, "DEV", corpus_id_set, doc_ids, rng, max_queries, max_pairs, include_negatives=False
        )
        if dev_queries:
            outputs["queries_dev.jsonl"] = ("jsonl", dev_queries)
            outputs["qrels_dev.tsv"] = ("qrels", dev_qrels)
    return outputs


def normalize_beir_corpus(source: Iterable[Mapping[str, Any]], max_docs: int | None) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    ids: set[str] = set()
    for row in source:
        if limit_reached(len(rows), max_docs):
            break
        doc_id = safe_text(first_value(row, "_id", "id", "docid", "corpus-id", "document-id"))
        text = safe_text(first_value(row, "text", "document", "contents", "body"))
        if not doc_id or not text or doc_id in ids:
            continue
        ids.add(doc_id)
        rows.append({"_id": doc_id, "title": safe_text(first_value(row, "title")), "text": text})
    if not rows:
        raise RuntimeError("BEIR corpus normalization produced no valid documents.")
    return rows, ids


def normalize_beir_queries(source: Iterable[Mapping[str, Any]], max_queries: int | None) -> tuple[list[dict[str, str]], set[str]]:
    rows: list[dict[str, str]] = []
    ids: set[str] = set()
    for row in source:
        if limit_reached(len(rows), max_queries):
            break
        query_id = safe_text(first_value(row, "_id", "id", "query-id", "query_id", "qid"))
        text = safe_text(first_value(row, "text", "title", "query", "question"))
        if not query_id or not text or query_id in ids:
            continue
        ids.add(query_id)
        rows.append({"_id": query_id, "text": text})
    return rows, ids


def normalize_beir_qrels(
    source: Iterable[Mapping[str, Any]], query_ids: set[str], corpus_ids: set[str]
) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for row in source:
        query_id = safe_text(first_value(row, "query-id", "query_id", "qid", "_id"))
        corpus_id = safe_text(first_value(row, "corpus-id", "corpus_id", "docid", "document-id", "doc_id"))
        score_value = first_value(row, "score", "relevance", "label")
        try:
            score = int(score_value)
        except (TypeError, ValueError):
            continue
        if query_id in query_ids and corpus_id in corpus_ids:
            rows.append((query_id, corpus_id, score))
    return rows


def available_hf_splits(dataset_id: str, cache_dir: Path) -> list[str]:
    """Return advertised splits when the installed datasets version supports discovery."""
    try:
        from datasets import get_dataset_split_names
    except ImportError:
        return []

    try:
        return list(get_dataset_split_names(dataset_id, cache_dir=str(cache_dir)))
    except TypeError:
        try:
            return list(get_dataset_split_names(dataset_id))
        except Exception:
            return []
    except Exception:
        return []


def load_beir_qrels(dataset_id: str, cache_dir: Path) -> tuple[Any, str]:
    """Load the most appropriate qrels split, tolerating dataset-version split differences."""
    split_names = available_hf_splits(dataset_id, cache_dir)
    if split_names:
        preferred = ("test", "validation", "dev", "train")
        lowered = {name.lower(): name for name in split_names}
        selected_split = next((lowered[name] for name in preferred if name in lowered), split_names[0])
        try:
            return load_dataset(dataset_id, split=selected_split, cache_dir=str(cache_dir)), selected_split
        except Exception as exc:
            raise RuntimeError(f"Unable to load BEIR qrels dataset '{dataset_id}' split '{selected_split}': {exc}") from exc

    try:
        loaded = load_dataset(dataset_id, cache_dir=str(cache_dir))
    except Exception as exc:
        raise RuntimeError(f"Unable to load BEIR qrels dataset '{dataset_id}': {exc}") from exc
    if hasattr(loaded, "keys"):
        available = list(loaded.keys())
        if not available:
            raise RuntimeError(f"BEIR qrels dataset '{dataset_id}' exposes no splits.")
        preferred = ("test", "validation", "dev", "train")
        lowered = {str(name).lower(): str(name) for name in available}
        selected_split = next((lowered[name] for name in preferred if name in lowered), str(available[0]))
        return loaded[selected_split], selected_split
    return loaded, "default"


def prepare_beir(
    config: Mapping[str, Any], cache_dir: Path, max_docs: int | None, max_queries: int | None
) -> dict[str, tuple[str, Any]]:
    require_keys(config, ("hf_corpus_config", "hf_queries_config", "hf_corpus_split", "hf_queries_split", "hf_qrels"))
    try:
        corpus_source = load_dataset(
            config["hf_dataset"],
            config["hf_corpus_config"],
            split=config["hf_corpus_split"],
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to load BEIR corpus '{config['hf_dataset']}' with configured corpus split: {exc}") from exc
    corpus_rows, corpus_ids = normalize_beir_corpus(corpus_source, max_docs)

    try:
        query_source = load_dataset(
            config["hf_dataset"],
            config["hf_queries_config"],
            split=config["hf_queries_split"],
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(f"Unable to load BEIR queries '{config['hf_dataset']}' with configured queries split: {exc}") from exc
    query_rows, query_ids = normalize_beir_queries(query_source, max_queries)
    if not query_rows:
        raise RuntimeError("BEIR query normalization produced no valid queries.")

    qrels_source, qrels_split = load_beir_qrels(config["hf_qrels"], cache_dir)
    qrel_rows = normalize_beir_qrels(qrels_source, query_ids, corpus_ids)
    if not qrel_rows:
        raise RuntimeError("BEIR qrels normalization produced no rows matching the normalized corpus and queries.")

    print(f"Using BEIR qrels split: {qrels_split}")
    outputs: dict[str, tuple[str, Any]] = {
        "corpus.jsonl": ("jsonl", corpus_rows),
        "queries_test.jsonl": ("jsonl", query_rows),
        "qrels_test.tsv": ("qrels", qrel_rows),
    }
    return outputs


def write_outputs(output_dir: Path, outputs: Mapping[str, tuple[str, Any]]) -> tuple[list[str], dict[str, int]]:
    created_files: list[str] = []
    counts: dict[str, int] = {}
    for filename, (kind, rows) in outputs.items():
        path = output_dir / filename
        if kind == "existing_jsonl":
            counts[filename] = count_jsonl(path)
        elif kind == "jsonl":
            write_jsonl(path, rows)
            counts[filename] = count_jsonl(path)
        elif kind == "qrels":
            write_tsv(path, QRELS_HEADER, rows)
            counts[filename] = count_tsv_rows(path)
        elif kind == "triples":
            write_tsv(path, TRIPLES_HEADER, rows)
            counts[filename] = count_tsv_rows(path)
        else:
            raise RuntimeError(f"Unknown output kind '{kind}' for {filename}.")
        created_files.append(filename)
    return created_files, counts


def build_dataset_info(
    config: Mapping[str, Any], created_files: list[str], counts: dict[str, int], args: argparse.Namespace
) -> dict[str, Any]:
    info = {
        "dataset_name": config["dataset_name"],
        "source": config["source"],
        "task": config["task"],
        "hf_dataset": config["hf_dataset"],
        "output_dir": config["output_dir"],
        "cache_dir": config["cache_dir"],
        "created_files": created_files + ["dataset_info.json"],
        "counts": counts,
        "random_seed": int(config.get("random_seed", 42)),
        "status": "complete",
    }
    if "hf_qrels" in config:
        info["hf_qrels"] = config["hf_qrels"]
    for name in ("max_docs", "max_queries", "max_pairs"):
        value = getattr(args, name)
        if value is not None:
            info[name] = value
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to a dataset YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing normalized files in the output directory.")
    parser.add_argument("--max-docs", type=int, help="Optional cap on normalized corpus documents.")
    parser.add_argument("--max-queries", type=int, help="Optional cap on normalized queries per split.")
    parser.add_argument("--max-pairs", type=int, help="Optional cap on inspected NQ320K pair rows per split.")
    args = parser.parse_args()

    if any(value is not None and value < 1 for value in (args.max_docs, args.max_queries, args.max_pairs)):
        parser.error("--max-docs, --max-queries, and --max-pairs must be positive integers when supplied.")

    try:
        config = load_yaml(args.config)
        require_keys(config, ("dataset_name", "source", "hf_dataset", "task", "output_dir", "cache_dir"))
        output_dir = ensure_dir(config["output_dir"])
        clean_normalized_outputs(output_dir, args.overwrite)
        cache_dir = set_hf_cache(config["cache_dir"])

        if config["dataset_name"] == "nq320k":
            outputs = prepare_nq320k(config, output_dir, cache_dir, args.max_docs, args.max_queries, args.max_pairs)
        elif config["dataset_name"] in {"beir_scifact", "beir_fiqa"}:
            outputs = prepare_beir(config, cache_dir, args.max_docs, args.max_queries)
        else:
            raise RuntimeError(f"No Phase 1 adapter is available for dataset '{config['dataset_name']}'.")

        created_files, counts = write_outputs(output_dir, outputs)
        info = build_dataset_info(config, created_files, counts, args)
        info_path = output_dir / "dataset_info.json"
        info_path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
        print(f"Normalized {config['dataset_name']} into {output_dir}")
        for filename in created_files:
            print(f"  {filename}: {counts[filename]} rows")
        print("  dataset_info.json: metadata")
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: unexpected data-preparation failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

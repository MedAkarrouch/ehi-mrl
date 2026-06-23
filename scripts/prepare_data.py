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


def dataset_mapping(dataset: Any) -> dict[str, Iterable[Mapping[str, Any]]]:
    """Represent a DatasetDict or single Dataset as named iterable splits."""
    if hasattr(dataset, "keys"):
        return {str(name): dataset[name] for name in dataset.keys()}
    return {"default": dataset}


def first_value(row: Mapping[str, Any], *names: str) -> Any:
    normalized = {str(key).lower().replace("_", "-"): value for key, value in row.items()}
    for name in names:
        key = name.lower().replace("_", "-")
        if key in normalized:
            return normalized[key]
    return None


def normalized_role(split_name: str) -> str:
    name = split_name.lower()
    if "train" in name:
        return "train"
    if "dev" in name or "valid" in name or "validation" in name:
        return "dev"
    return "test"


def choose_split(splits: Mapping[str, Any], preferred: tuple[str, ...]) -> Any | None:
    lowered = {name.lower(): value for name, value in splits.items()}
    for candidate in preferred:
        if candidate in lowered:
            return lowered[candidate]
    for name, value in splits.items():
        if any(candidate in name.lower() for candidate in preferred):
            return value
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


def load_beir_main_splits(dataset_id: str, cache_dir: Path) -> tuple[Any, dict[str, Iterable[Mapping[str, Any]]]]:
    try:
        loaded = load_dataset(dataset_id, cache_dir=str(cache_dir))
        return loaded, dataset_mapping(loaded)
    except Exception as exc:
        raise RuntimeError(f"Unable to load BEIR dataset '{dataset_id}': {exc}") from exc


def fallback_beir_split(dataset_id: str, cache_dir: Path, config_name: str, split_names: tuple[str, ...]) -> Any | None:
    for split_name in split_names:
        try:
            return load_dataset(dataset_id, config_name, split=split_name, cache_dir=str(cache_dir))
        except Exception:
            continue
    return None


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
        text = safe_text(first_value(row, "text", "query", "question"))
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


def prepare_beir(
    config: Mapping[str, Any], cache_dir: Path, max_docs: int | None, max_queries: int | None
) -> dict[str, tuple[str, Any]]:
    require_keys(config, ("hf_qrels",))
    _loaded, main_splits = load_beir_main_splits(config["hf_dataset"], cache_dir)
    corpus_source = choose_split(main_splits, ("corpus", "documents", "docs"))
    if corpus_source is None:
        corpus_source = fallback_beir_split(config["hf_dataset"], cache_dir, "corpus", ("corpus", "train"))
    if corpus_source is None:
        raise RuntimeError("Could not detect a BEIR corpus split. Inspect the Hugging Face dataset schema and update the adapter.")
    corpus_rows, corpus_ids = normalize_beir_corpus(corpus_source, max_docs)

    query_sources: dict[str, Iterable[Mapping[str, Any]]] = {}
    for name, split in main_splits.items():
        if split is corpus_source or any(token in name.lower() for token in ("corpus", "document", "docs")):
            continue
        query_sources[normalized_role(name)] = split
    if not query_sources:
        fallback_queries = fallback_beir_split(config["hf_dataset"], cache_dir, "queries", ("queries", "test", "train"))
        if fallback_queries is not None:
            query_sources["test"] = fallback_queries

    query_rows: dict[str, list[dict[str, str]]] = {}
    query_ids: dict[str, set[str]] = {}
    for role, source in query_sources.items():
        rows, ids = normalize_beir_queries(source, max_queries)
        if rows:
            query_rows[role] = rows
            query_ids[role] = ids
    if not query_rows:
        raise RuntimeError("BEIR query normalization produced no valid queries.")

    try:
        qrels_loaded = load_dataset(config["hf_qrels"], cache_dir=str(cache_dir))
    except Exception as exc:
        raise RuntimeError(f"Unable to load BEIR qrels dataset '{config['hf_qrels']}': {exc}") from exc
    qrels_splits = dataset_mapping(qrels_loaded)
    qrel_rows: dict[str, list[tuple[str, str, int]]] = {}
    for split_name, source in qrels_splits.items():
        role = normalized_role(split_name)
        if role not in query_ids and len(query_ids) == 1:
            role = next(iter(query_ids))
        if role not in query_ids:
            continue
        rows = normalize_beir_qrels(source, query_ids[role], corpus_ids)
        if rows:
            qrel_rows.setdefault(role, []).extend(rows)
    if not qrel_rows:
        raise RuntimeError("BEIR qrels normalization produced no rows matching the normalized corpus and queries.")

    outputs: dict[str, tuple[str, Any]] = {"corpus.jsonl": ("jsonl", corpus_rows)}
    for role, rows in query_rows.items():
        outputs[f"queries_{role}.jsonl"] = ("jsonl", rows)
    for role, rows in qrel_rows.items():
        outputs[f"qrels_{role}.tsv"] = ("qrels", rows)
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

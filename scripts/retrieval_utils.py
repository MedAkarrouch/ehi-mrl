"""Reusable helpers for Phase 2 dense retrieval baselines."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from data_utils import ensure_dir, iter_jsonl, load_yaml, safe_text


QRELS_HEADER = ["query-id", "corpus-id", "score"]
RUN_HEADER = ["query-id", "corpus-id", "score", "rank"]
DEFAULT_RETRIEVAL_METRICS = ("Hit@1", "MRR@10", "Recall@1", "Recall@10", "Recall@100", "nDCG@10")


def parse_simple_yaml_scalar(value: str) -> Any:
    cleaned = value.strip()
    if not cleaned:
        return ""
    if cleaned.lower() == "null":
        return None
    if cleaned.startswith("[") and cleaned.endswith("]"):
        inner = cleaned[1:-1].strip()
        if not inner:
            return []
        return [parse_simple_yaml_scalar(item.strip()) for item in inner.split(",")]
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        return cleaned[1:-1]
    lowered = cleaned.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(cleaned)
    except ValueError:
        pass
    try:
        return float(cleaned)
    except ValueError:
        return cleaned


def load_simple_yaml_mapping(path: Path) -> dict[str, Any]:
    """Parse the top-level mapping/list subset used by project configs.

    PyYAML is available on HPC, but Codex-safe local tests should not require
    extra packages. This fallback supports the simple config structure used by
    this repository: top-level scalars and top-level ``- item`` lists.
    """
    data: dict[str, Any] = {}
    current_list_key: str | None = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if line[:1].isspace():
                    if current_list_key and stripped.startswith("- "):
                        data[current_list_key].append(parse_simple_yaml_scalar(stripped[2:]))
                        continue
                    raise RuntimeError(f"Invalid simple YAML in '{path}' at line {line_number}: unsupported nesting.")
                current_list_key = None
                if ":" not in stripped:
                    raise RuntimeError(f"Invalid simple YAML in '{path}' at line {line_number}: expected key: value.")
                key, value = stripped.split(":", 1)
                key = key.strip()
                if not key:
                    raise RuntimeError(f"Invalid simple YAML in '{path}' at line {line_number}: empty key.")
                if value.strip():
                    data[key] = parse_simple_yaml_scalar(value)
                else:
                    data[key] = []
                    current_list_key = key
    except OSError as exc:
        raise RuntimeError(f"Could not read YAML file '{path}': {exc}") from exc
    if not data:
        raise RuntimeError(f"YAML file '{path}' must contain a mapping.")
    return data


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    try:
        return load_yaml(config_path)
    except RuntimeError as exc:
        if "PyYAML is required" not in str(exc):
            raise
        return load_simple_yaml_mapping(config_path)


def repo_root_from_script(script_file: str) -> Path:
    return Path(script_file).resolve().parents[1]


def resolve_path(repo_root: Path, path: str | Path) -> Path:
    target = Path(path)
    if target.is_absolute():
        return target
    return repo_root / target


def build_document_text(row: Mapping[str, Any]) -> str:
    title = safe_text(row.get("title"))
    text = safe_text(row.get("text"))
    if title:
        return f"{title}\n{text}" if text else title
    return text


def load_corpus_jsonl(path: str | Path, max_docs: int | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for row_index, row in enumerate(iter_jsonl(path), start=1):
        if max_docs is not None and len(rows) >= max_docs:
            break
        doc_id = safe_text(row.get("_id"))
        if not doc_id:
            raise RuntimeError(f"{Path(path).name}:{row_index} has an empty _id.")
        rows.append((doc_id, build_document_text(row)))
    return rows


def load_queries_jsonl(path: str | Path, max_queries: int | None = None) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for row_index, row in enumerate(iter_jsonl(path), start=1):
        if max_queries is not None and len(rows) >= max_queries:
            break
        query_id = safe_text(row.get("_id"))
        if not query_id:
            raise RuntimeError(f"{Path(path).name}:{row_index} has an empty _id.")
        rows.append((query_id, safe_text(row.get("text"))))
    return rows


def load_qrels_tsv(path: str | Path) -> dict[str, dict[str, float]]:
    qrels_path = Path(path)
    qrels: dict[str, dict[str, float]] = {}
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"{qrels_path.name} is empty; expected header {QRELS_HEADER}.") from exc
        if header != QRELS_HEADER:
            raise RuntimeError(f"{qrels_path.name} header must be {QRELS_HEADER}; found {header}.")
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 3:
                raise RuntimeError(f"{qrels_path.name}:{line_number} must contain exactly 3 columns.")
            query_id, doc_id, score_text = (safe_text(value) for value in row)
            if not query_id or not doc_id:
                raise RuntimeError(f"{qrels_path.name}:{line_number} has an empty query or corpus id.")
            try:
                score = float(score_text)
            except ValueError as exc:
                raise RuntimeError(f"{qrels_path.name}:{line_number} score '{score_text}' is not numeric.") from exc
            qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def load_run_tsv(path: str | Path) -> dict[str, dict[str, float]]:
    run_path = Path(path)
    rows_by_query: dict[str, list[tuple[int, str, float]]] = {}
    with run_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"{run_path.name} is empty; expected header {RUN_HEADER}.") from exc
        if header != RUN_HEADER:
            raise RuntimeError(f"{run_path.name} header must be {RUN_HEADER}; found {header}.")
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != 4:
                raise RuntimeError(f"{run_path.name}:{line_number} must contain exactly 4 columns.")
            query_id, doc_id, score_text, rank_text = (safe_text(value) for value in row)
            try:
                score = float(score_text)
                rank = int(rank_text)
            except ValueError as exc:
                raise RuntimeError(f"{run_path.name}:{line_number} has a non-numeric score or rank.") from exc
            rows_by_query.setdefault(query_id, []).append((rank, doc_id, score))
    run: dict[str, dict[str, float]] = {}
    for query_id, rows in rows_by_query.items():
        run[query_id] = {doc_id: score for rank, doc_id, score in sorted(rows, key=lambda item: item[0])}
    return run


def write_run_tsv(path: str | Path, rankings: Mapping[str, Sequence[tuple[str, float]]]) -> None:
    run_path = Path(path)
    ensure_dir(run_path.parent)
    with run_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(RUN_HEADER)
        for query_id, docs in rankings.items():
            for rank, (doc_id, score) in enumerate(docs, start=1):
                writer.writerow([query_id, doc_id, f"{float(score):.8f}", rank])


def save_id_list(path: str | Path, ids: Sequence[str]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(list(ids), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_id_list(path: str | Path) -> list[str]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list) or not all(isinstance(value, str) for value in data):
        raise RuntimeError(f"{source.name} must contain a JSON list of strings.")
    return data


def detect_split_from_filename(path: str | Path) -> str:
    stem = Path(path).stem
    for prefix in ("queries_", "qrels_", "run_", "metrics_"):
        if stem.startswith(prefix):
            return stem.removeprefix(prefix)
    raise RuntimeError(f"Could not detect split from filename: {Path(path).name}")


def filter_queries_to_qrels_covered(
    queries: Sequence[tuple[str, str]], qrels: Mapping[str, Mapping[str, float]]
) -> list[tuple[str, str]]:
    return [(query_id, text) for query_id, text in queries if query_id in qrels]


def qrels_query_ids_missing_from_queries(
    queries: Sequence[tuple[str, str]], qrels: Mapping[str, Mapping[str, float]]
) -> list[str]:
    query_ids = {query_id for query_id, _text in queries}
    return sorted(set(qrels) - query_ids)


def dcg_at_k(ranked_doc_ids: Sequence[str], relevant_scores: Mapping[str, float], k: int) -> float:
    total = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        relevance = max(0.0, float(relevant_scores.get(doc_id, 0.0)))
        if relevance <= 0:
            continue
        total += (math.pow(2.0, relevance) - 1.0) / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_doc_ids: Sequence[str], relevant_scores: Mapping[str, float], k: int) -> float:
    ideal_relevances = sorted((score for score in relevant_scores.values() if score > 0), reverse=True)[:k]
    ideal_doc_ids = [f"__ideal_{index}" for index, _score in enumerate(ideal_relevances)]
    ideal_scores = dict(zip(ideal_doc_ids, ideal_relevances))
    ideal_dcg = dcg_at_k(ideal_doc_ids, ideal_scores, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_doc_ids, relevant_scores, k) / ideal_dcg


def compute_retrieval_metrics(
    qrels: Mapping[str, Mapping[str, float]],
    run: Mapping[str, Mapping[str, float]],
    evaluated_query_ids: Iterable[str],
    metric_names: Sequence[str] | None = None,
) -> dict[str, float]:
    requested_metrics = tuple(metric_names or DEFAULT_RETRIEVAL_METRICS)
    unsupported = [metric_name for metric_name in requested_metrics if metric_name not in DEFAULT_RETRIEVAL_METRICS]
    if unsupported:
        raise RuntimeError(f"Unsupported retrieval metric(s): {', '.join(unsupported)}")
    metric_sums = {metric_name: 0.0 for metric_name in requested_metrics}
    query_count = 0
    for query_id in evaluated_query_ids:
        relevant_scores = {doc_id: score for doc_id, score in qrels.get(query_id, {}).items() if score > 0}
        relevant_doc_ids = set(relevant_scores)
        ranked_doc_ids = list(run.get(query_id, {}).keys())
        query_count += 1

        if "Hit@1" in metric_sums and ranked_doc_ids[:1] and ranked_doc_ids[0] in relevant_doc_ids:
            metric_sums["Hit@1"] += 1.0

        if "MRR@10" in metric_sums:
            for rank, doc_id in enumerate(ranked_doc_ids[:10], start=1):
                if doc_id in relevant_doc_ids:
                    metric_sums["MRR@10"] += 1.0 / rank
                    break

        for cutoff in (1, 10, 100):
            metric_name = f"Recall@{cutoff}"
            if metric_name in metric_sums and relevant_doc_ids:
                found = len(set(ranked_doc_ids[:cutoff]) & relevant_doc_ids)
                metric_sums[metric_name] += found / len(relevant_doc_ids)

        if "nDCG@10" in metric_sums:
            metric_sums["nDCG@10"] += ndcg_at_k(ranked_doc_ids, relevant_scores, 10)

    if query_count == 0:
        return {name: 0.0 for name in metric_sums}
    return {name: value / query_count for name, value in metric_sums.items()}

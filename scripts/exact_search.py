#!/usr/bin/env python3
"""Run chunked exact dense retrieval over saved normalized embeddings."""

from __future__ import annotations

import argparse
import json
import ast
import struct
import sys
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

from data_utils import ensure_dir
from retrieval_utils import load_config, load_id_list, repo_root_from_script, resolve_path, write_run_tsv


def import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def select_device(requested_device: str) -> tuple[str, Any | None]:
    torch = import_torch()
    requested = requested_device.lower()
    if requested == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda", torch
        return "cpu", torch
    if requested.startswith("cuda"):
        if torch is None:
            raise RuntimeError("CUDA was requested, but torch is not installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
        return requested_device, torch
    if requested == "cpu":
        return "cpu", torch
    raise RuntimeError(f"Unsupported device setting: {requested_device}")


def cuda_info(torch: Any | None, device: str) -> dict[str, Any]:
    if torch is None or not device.startswith("cuda") or not torch.cuda.is_available():
        return {}
    props = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "total_gpu_memory_gb": round(props.total_memory / 1024**3, 2),
    }


def print_cuda_startup(torch: Any | None, device: str) -> None:
    info = cuda_info(torch, device)
    if info:
        print(f"Using CUDA device: {info['gpu_name']}")
        print(f"total vram gb: {info['total_gpu_memory_gb']}")
        torch.cuda.reset_peak_memory_stats()


def array_shape(array: Any) -> tuple[int, int]:
    if np is not None and hasattr(array, "shape"):
        if len(array.shape) != 2:
            return (-1, -1)
        return int(array.shape[0]), int(array.shape[1])
    if not isinstance(array, list) or (array and not isinstance(array[0], list)):
        return (-1, -1)
    return len(array), len(array[0]) if array else 0


def load_npy_float32_2d(path: Path) -> list[list[float]]:
    """Minimal NumPy .npy v1/v2 float32 reader for Codex offline tests."""
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise RuntimeError(f"{path.name} is not a NumPy .npy file.")
        major, _minor = handle.read(2)
        if major == 1:
            header_length = struct.unpack("<H", handle.read(2))[0]
        elif major == 2:
            header_length = struct.unpack("<I", handle.read(4))[0]
        else:
            raise RuntimeError(f"{path.name} uses unsupported .npy version {major}.")
        header = ast.literal_eval(handle.read(header_length).decode("latin1").strip())
        if header.get("descr") not in {"<f4", "|f4"} or header.get("fortran_order"):
            raise RuntimeError(f"{path.name} fallback reader supports only little-endian C-order float32 arrays.")
        shape = tuple(header.get("shape", ()))
        if len(shape) != 2:
            raise RuntimeError(f"{path.name} fallback reader supports only 2D arrays.")
        row_count, column_count = int(shape[0]), int(shape[1])
        raw = handle.read()
    expected_bytes = row_count * column_count * 4
    if len(raw) != expected_bytes:
        raise RuntimeError(f"{path.name} has {len(raw)} data bytes; expected {expected_bytes}.")
    values = struct.unpack("<" + "f" * (row_count * column_count), raw)
    return [
        list(values[row_index * column_count : (row_index + 1) * column_count])
        for row_index in range(row_count)
    ]


def load_embedding_array(path: Path) -> Any:
    if np is not None:
        return np.load(path).astype(np.float32, copy=False)
    return load_npy_float32_2d(path)


def validate_embeddings(query_embeddings: Any, corpus_embeddings: Any, query_ids: list[str], corpus_ids: list[str]) -> None:
    query_shape = array_shape(query_embeddings)
    corpus_shape = array_shape(corpus_embeddings)
    if query_shape[0] < 0 or corpus_shape[0] < 0:
        raise RuntimeError("Query and corpus embeddings must be 2D arrays.")
    if query_shape[1] != corpus_shape[1]:
        raise RuntimeError(
            f"Embedding dimensions differ: queries={query_shape[1]}, corpus={corpus_shape[1]}."
        )
    if query_shape[0] != len(query_ids):
        raise RuntimeError("Number of query embeddings does not match query ID count.")
    if corpus_shape[0] != len(corpus_ids):
        raise RuntimeError("Number of corpus embeddings does not match corpus ID count.")
    if not query_ids:
        raise RuntimeError("No query embeddings were loaded.")
    if not corpus_ids:
        raise RuntimeError("No corpus embeddings were loaded.")


def search_numpy(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    top_k: int,
    query_batch_size: int,
    corpus_chunk_size: int,
) -> dict[str, list[tuple[str, float]]]:
    if np is None:
        raise RuntimeError("NumPy search was requested but NumPy is not installed.")
    actual_k = min(top_k, len(corpus_ids))
    rankings: dict[str, list[tuple[str, float]]] = {}
    for query_start in range(0, len(query_ids), query_batch_size):
        query_end = min(query_start + query_batch_size, len(query_ids))
        query_batch = query_embeddings[query_start:query_end].astype(np.float32, copy=False)
        best_scores = np.full((query_batch.shape[0], actual_k), -np.inf, dtype=np.float32)
        best_indices = np.full((query_batch.shape[0], actual_k), -1, dtype=np.int64)
        for corpus_start in range(0, len(corpus_ids), corpus_chunk_size):
            corpus_end = min(corpus_start + corpus_chunk_size, len(corpus_ids))
            corpus_chunk = corpus_embeddings[corpus_start:corpus_end].astype(np.float32, copy=False)
            scores = query_batch @ corpus_chunk.T
            local_k = min(actual_k, scores.shape[1])
            local_indices = np.argpartition(-scores, kth=local_k - 1, axis=1)[:, :local_k]
            local_scores = np.take_along_axis(scores, local_indices, axis=1)
            local_indices = local_indices + corpus_start
            combined_scores = np.concatenate([best_scores, local_scores], axis=1)
            combined_indices = np.concatenate([best_indices, local_indices], axis=1)
            keep_indices = np.argpartition(-combined_scores, kth=actual_k - 1, axis=1)[:, :actual_k]
            best_scores = np.take_along_axis(combined_scores, keep_indices, axis=1)
            best_indices = np.take_along_axis(combined_indices, keep_indices, axis=1)
        for row_offset, query_id in enumerate(query_ids[query_start:query_end]):
            order = np.lexsort((best_indices[row_offset], -best_scores[row_offset]))
            rankings[query_id] = [
                (corpus_ids[int(best_indices[row_offset, index])], float(best_scores[row_offset, index]))
                for index in order
                if best_indices[row_offset, index] >= 0
            ]
    return rankings


def search_python(
    query_embeddings: list[list[float]],
    corpus_embeddings: list[list[float]],
    query_ids: list[str],
    corpus_ids: list[str],
    top_k: int,
    query_batch_size: int,
    corpus_chunk_size: int,
) -> dict[str, list[tuple[str, float]]]:
    """Small stdlib fallback used only by local fake-data tests without NumPy."""
    actual_k = min(top_k, len(corpus_ids))
    rankings: dict[str, list[tuple[str, float]]] = {}
    for query_start in range(0, len(query_ids), query_batch_size):
        query_end = min(query_start + query_batch_size, len(query_ids))
        for local_query_index, query_vector in enumerate(query_embeddings[query_start:query_end]):
            scored_docs: list[tuple[str, float]] = []
            for corpus_start in range(0, len(corpus_ids), corpus_chunk_size):
                corpus_end = min(corpus_start + corpus_chunk_size, len(corpus_ids))
                for corpus_id, corpus_vector in zip(
                    corpus_ids[corpus_start:corpus_end],
                    corpus_embeddings[corpus_start:corpus_end],
                ):
                    score = sum(query_value * corpus_value for query_value, corpus_value in zip(query_vector, corpus_vector))
                    scored_docs.append((corpus_id, float(score)))
                scored_docs = sorted(scored_docs, key=lambda item: (-item[1], item[0]))[:actual_k]
            rankings[query_ids[query_start + local_query_index]] = scored_docs
    return rankings


def search_torch(
    torch: Any,
    device: str,
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    top_k: int,
    query_batch_size: int,
    corpus_chunk_size: int,
) -> dict[str, list[tuple[str, float]]]:
    if np is None:
        raise RuntimeError("Torch search path requires NumPy arrays; use the stdlib fallback for local tests.")
    device_obj = torch.device(device)
    actual_k = min(top_k, len(corpus_ids))
    rankings: dict[str, list[tuple[str, float]]] = {}
    with torch.no_grad():
        for query_start in range(0, len(query_ids), query_batch_size):
            query_end = min(query_start + query_batch_size, len(query_ids))
            query_batch = torch.as_tensor(
                query_embeddings[query_start:query_end].astype(np.float32, copy=False), device=device_obj
            )
            best_scores = torch.full((query_batch.shape[0], actual_k), -float("inf"), dtype=torch.float32, device=device_obj)
            best_indices = torch.full((query_batch.shape[0], actual_k), -1, dtype=torch.long, device=device_obj)
            for corpus_start in range(0, len(corpus_ids), corpus_chunk_size):
                corpus_end = min(corpus_start + corpus_chunk_size, len(corpus_ids))
                corpus_chunk = torch.as_tensor(
                    corpus_embeddings[corpus_start:corpus_end].astype(np.float32, copy=False), device=device_obj
                )
                scores = query_batch @ corpus_chunk.T
                local_k = min(actual_k, scores.shape[1])
                local_scores, local_indices = torch.topk(scores, k=local_k, dim=1)
                local_indices = local_indices + corpus_start
                combined_scores = torch.cat([best_scores, local_scores], dim=1)
                combined_indices = torch.cat([best_indices, local_indices], dim=1)
                best_scores, selected = torch.topk(combined_scores, k=actual_k, dim=1)
                best_indices = torch.gather(combined_indices, 1, selected)
            best_scores_cpu = best_scores.cpu().numpy()
            best_indices_cpu = best_indices.cpu().numpy()
            for row_offset, query_id in enumerate(query_ids[query_start:query_end]):
                order = np.lexsort((best_indices_cpu[row_offset], -best_scores_cpu[row_offset]))
                rankings[query_id] = [
                    (corpus_ids[int(best_indices_cpu[row_offset, index])], float(best_scores_cpu[row_offset, index]))
                    for index in order
                    if best_indices_cpu[row_offset, index] >= 0
                ]
    return rankings


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "cuda" in message and ("out of memory" in message or "cuda oom" in message)


def write_json(path: Path, data: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to an exact baseline YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run/search outputs.")
    parser.add_argument("--query-batch-size", type=int, help="Override query batch size.")
    parser.add_argument("--corpus-chunk-size", type=int, help="Override corpus chunk size.")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, cuda:0.")
    args = parser.parse_args()

    current_query_batch_size: int | None = None
    current_corpus_chunk_size: int | None = None
    try:
        repo_root = repo_root_from_script(__file__)
        config = load_config(resolve_path(repo_root, args.config))
        if str(config.get("similarity", "cosine")).lower() != "cosine":
            raise RuntimeError("Phase 2 exact search only supports cosine similarity via normalized dot product.")

        split = str(config["split"])
        embedding_dir = resolve_path(repo_root, config["embedding_dir"])
        results_dir = resolve_path(repo_root, config["results_dir"])
        run_path = results_dir / f"run_{split}.tsv"
        info_path = results_dir / "search_info.json"
        if not args.overwrite:
            existing = [path for path in (run_path, info_path) if path.exists()]
            if existing:
                names = ", ".join(str(path) for path in existing)
                raise RuntimeError(f"Refusing to overwrite existing search output(s): {names}. Use --overwrite.")

        top_k = int(config.get("top_k", 100))
        query_batch_size = int(args.query_batch_size or config.get("query_batch_size", 128))
        corpus_chunk_size = int(args.corpus_chunk_size or config.get("corpus_chunk_size", 50000))
        current_query_batch_size = query_batch_size
        current_corpus_chunk_size = corpus_chunk_size
        if top_k <= 0 or query_batch_size <= 0 or corpus_chunk_size <= 0:
            raise RuntimeError("top_k, query_batch_size, and corpus_chunk_size must be positive.")

        query_embeddings = load_embedding_array(embedding_dir / f"queries_{split}_embeddings.npy")
        corpus_embeddings = load_embedding_array(embedding_dir / "corpus_embeddings.npy")
        query_ids = load_id_list(embedding_dir / f"queries_{split}_ids.json")
        corpus_ids = load_id_list(embedding_dir / "corpus_ids.json")
        validate_embeddings(query_embeddings, corpus_embeddings, query_ids, corpus_ids)

        requested_device = str(args.device or config.get("device", "auto"))
        device, torch = select_device(requested_device)
        print_cuda_startup(torch, device)
        print(
            f"Running exact search: queries={len(query_ids)} corpus={len(corpus_ids)} "
            f"top_k={top_k} query_batch_size={query_batch_size} corpus_chunk_size={corpus_chunk_size} device={device}"
        )
        if torch is not None and np is not None:
            rankings = search_torch(
                torch,
                device,
                query_embeddings,
                corpus_embeddings,
                query_ids,
                corpus_ids,
                top_k,
                query_batch_size,
                corpus_chunk_size,
            )
        elif np is not None:
            print("warning: torch is not installed; using NumPy CPU fallback for local testing.", file=sys.stderr)
            rankings = search_numpy(
                query_embeddings, corpus_embeddings, query_ids, corpus_ids, top_k, query_batch_size, corpus_chunk_size
            )
        else:
            print("warning: torch and numpy are not installed; using stdlib fallback for tiny local tests.", file=sys.stderr)
            rankings = search_python(
                query_embeddings, corpus_embeddings, query_ids, corpus_ids, top_k, query_batch_size, corpus_chunk_size
            )

        write_run_tsv(run_path, rankings)
        info: dict[str, Any] = {
            "dataset_name": config["dataset_name"],
            "split": split,
            "top_k": top_k,
            "similarity": config.get("similarity", "cosine"),
            "query_count": len(query_ids),
            "corpus_count": len(corpus_ids),
            "device_used": device,
            "query_batch_size": query_batch_size,
            "corpus_chunk_size": corpus_chunk_size,
        }
        info.update(cuda_info(torch, device))
        if torch is not None and device.startswith("cuda") and torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
            print("peak cuda memory allocated GB:", peak)
            info["peak_cuda_memory_allocated_gb"] = peak
        write_json(info_path, info)
        print(f"Wrote run file: {run_path}")
        return 0
    except RuntimeError as exc:
        if is_cuda_oom(exc):
            print("error: CUDA out of memory during exact search.", file=sys.stderr)
            print(f"query_batch_size={current_query_batch_size or 'unknown'}", file=sys.stderr)
            print(f"corpus_chunk_size={current_corpus_chunk_size or 'unknown'}", file=sys.stderr)
            print("Lower --query-batch-size, lower --corpus-chunk-size, or both.", file=sys.stderr)
            print("If the job landed on L40S by accident, use the H200-preferred Slurm job.", file=sys.stderr)
            return 1
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

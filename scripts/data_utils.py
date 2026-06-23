"""Small, dependency-light helpers for normalized retrieval data files."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and return it as a dictionary."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install project dependencies first.") from exc

    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise RuntimeError(f"Could not read YAML file '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in '{config_path}': {exc}") from exc

    if not isinstance(data, dict):
        raise RuntimeError(f"YAML file '{config_path}' must contain a mapping.")
    return data


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def safe_text(value: Any) -> str:
    """Return a stripped string, converting missing values to an empty string."""
    if value is None:
        return ""
    return str(value).strip()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in '{source}' at line {line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row in '{source}' at line {line_number} is not an object.")
            yield row


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def write_tsv(path: str | Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_tsv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    source = Path(path)
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            return [], []
        return header, [row for row in reader if row]


def file_exists_nonempty(path: str | Path) -> bool:
    target = Path(path)
    return target.is_file() and target.stat().st_size > 0


def count_jsonl(path: str | Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def count_tsv_rows(path: str | Path) -> int:
    _header, rows = read_tsv(path)
    return len(rows)


def set_hf_cache(cache_dir: str | Path) -> Path:
    """Set process-local Hugging Face cache variables and create the cache layout."""
    root = ensure_dir(Path(cache_dir).resolve())
    datasets_cache = ensure_dir(root / "datasets")
    transformers_cache = ensure_dir(root / "transformers")
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    return root

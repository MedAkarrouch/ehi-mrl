#!/usr/bin/env python3
"""Orchestrate Phase 2 embedding, exact search, and evaluation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from retrieval_utils import repo_root_from_script, resolve_path


def run_step(name: str, command: list[str], cwd: Path) -> None:
    print("")
    print(f"=== {name} ===")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Path to an exact baseline YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite embedding and search outputs.")
    parser.add_argument("--skip-embedding", action="store_true", help="Skip embedding generation.")
    parser.add_argument("--skip-search", action="store_true", help="Skip exact search.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip run evaluation.")
    parser.add_argument("--max-docs", type=int, help="Optional debug limit passed to embedding.")
    parser.add_argument("--max-queries", type=int, help="Optional debug limit passed to embedding.")
    args = parser.parse_args()

    try:
        repo_root = repo_root_from_script(__file__)
        config_path = resolve_path(repo_root, args.config)

        if not args.skip_embedding:
            command = [sys.executable, str(repo_root / "scripts" / "embed_dataset.py"), "--config", str(config_path)]
            if args.overwrite:
                command.append("--overwrite")
            if args.max_docs is not None:
                command.extend(["--max-docs", str(args.max_docs)])
            if args.max_queries is not None:
                command.extend(["--max-queries", str(args.max_queries)])
            run_step("Embedding", command, repo_root)

        if not args.skip_search:
            command = [sys.executable, str(repo_root / "scripts" / "exact_search.py"), "--config", str(config_path)]
            if args.overwrite:
                command.append("--overwrite")
            run_step("Exact search", command, repo_root)

        if not args.skip_eval:
            command = [sys.executable, str(repo_root / "scripts" / "evaluate_run.py"), "--config", str(config_path)]
            run_step("Evaluation", command, repo_root)

        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

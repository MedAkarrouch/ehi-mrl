#!/usr/bin/env python3
"""Offline checks for H200 dense batch benchmark robustness updates."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    script_path = ROOT / "scripts" / "benchmark_dense_batch_size.py"
    help_result = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--num-workers" in help_result.stdout
    assert "--prefetch-factor" in help_result.stdout
    assert "--persistent-workers" in help_result.stdout

    script_source = script_path.read_text(encoding="utf-8")
    assert "initialize_csv(output_path)" in script_source
    assert "write_rows(output_path, rows)" in script_source
    assert "del dataloader" in script_source
    assert "del optimizer" in script_source
    assert "cleanup_cuda(torch)" in script_source
    assert "default=2, help=\"Benchmark DataLoader worker count.\"" in script_source
    assert "default=2, help=\"Benchmark DataLoader prefetch factor.\"" in script_source
    assert "default=False, help=\"Use persistent DataLoader workers.\"" in script_source

    highmem_job = ROOT / "jobs" / "benchmark_dense_batch_h200_highmem.sbatch"
    assert highmem_job.is_file(), "Missing high-memory H200 benchmark job."
    highmem_text = highmem_job.read_text(encoding="utf-8")
    assert "#SBATCH --mem=200G" in highmem_text
    assert "--batch-sizes 1024,1280,1536,1792,2048" in highmem_text
    assert "dense_nq320k_h200_batch_sweep_highmem.csv" in highmem_text
    assert "--num-workers 2" in highmem_text
    assert "--prefetch-factor 2" in highmem_text
    assert "--persistent-workers false" in highmem_text

    train_config = (ROOT / "configs" / "train_dense_nq320k_distilbert.yaml").read_text(encoding="utf-8")
    assert "per_device_train_batch_size: 512" in train_config
    print("Dense benchmark infrastructure checks passed.")


if __name__ == "__main__":
    main()

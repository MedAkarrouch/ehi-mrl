#!/usr/bin/env python3
"""Fake tensor tests for dense modeling helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> None:
    if importlib.util.find_spec("torch") is None:
        print("Dense modeling fake tensor checks skipped: torch is not installed locally.")
        return
    import torch
    from dense_modeling import in_batch_contrastive_loss, l2_normalize, mean_pool

    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]], [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0]]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pooled = mean_pool(hidden, mask)
    assert pooled.shape == (2, 2)
    assert torch.allclose(pooled[0], torch.tensor([2.0, 2.0]))
    assert torch.allclose(pooled[1], torch.tensor([4.0, 0.0]))
    normalized = l2_normalize(pooled)
    assert torch.allclose(torch.linalg.norm(normalized, dim=1), torch.ones(2))
    loss = in_batch_contrastive_loss(normalized, normalized, temperature=0.05)
    assert loss.ndim == 0
    print("Dense modeling fake tensor checks passed.")


if __name__ == "__main__":
    main()

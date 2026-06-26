#!/usr/bin/env python3
"""Check paper-facing plot labels use method names, not internal phase names."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = ROOT / "scripts" / "plot_faiss_ivf_sweeps.py"


def main() -> None:
    source = PLOT_SCRIPT.read_text(encoding="utf-8")
    assert "Phase 2 exact" not in source
    assert "Phase 2 baseline" not in source
    assert "Phase 3 FAISS" not in source
    assert 'EXACT_METHOD_LABEL = "SBERT + Exact Search"' in source
    assert 'IVF_METHOD_LABEL = "SBERT + FAISS-IVF"' in source
    assert "label=EXACT_METHOD_LABEL" in source
    assert "IVF_METHOD_LABEL" in source
    print("Plot method-label checks passed.")


if __name__ == "__main__":
    main()

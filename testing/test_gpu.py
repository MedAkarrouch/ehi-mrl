#!/usr/bin/env python3
"""Small HPC-only CUDA smoke test. Do not run this in Codex."""

import platform
import sys

import torch


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {platform.python_version()}")
    print(f"torch version: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    print(f"device count: {torch.cuda.device_count()}")

    if not cuda_available:
        print("error: CUDA is not available; run this HPC-only test inside a GPU allocation.", file=sys.stderr)
        raise SystemExit(1)

    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    print(f"bf16 support: {bf16_supported}")
    tensor = torch.tensor([1.0, 2.0], device="cuda")
    result = tensor * 2
    assert torch.allclose(result.cpu(), torch.tensor([2.0, 4.0]))
    print("CUDA tensor smoke test passed.")


if __name__ == "__main__":
    main()

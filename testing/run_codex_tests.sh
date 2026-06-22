#!/usr/bin/env bash
set -euo pipefail

echo "Running Codex-safe offline tests..."
python testing/test_configs.py
python testing/test_scripts_compile.py
python testing/test_hpc_env_file.py
echo "Codex-safe tests passed. GPU/HPC tests were not run."
